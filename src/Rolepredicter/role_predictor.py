"""
Role Predictor - Werewolf Game Role Prediction System

This module provides the RolePredictor class for training and predicting roles
in Werewolf games using XGBoost with perspective-specific models.

The system supports four perspectives:
- Villager perspective
- Possessed perspective
- Seer (divination-based perspective)
- Werewolf (evil team perspective)

Features:
- Hyperparameter optimization via Optuna
- Constraint-based role assignment
- Multiple model management
- Easy prediction interface

Example:
    >>> # Initialize and train
    >>> predictor = RolePredictor('data.csv')
    >>> predictor.train(n_trials=200)
    
    >>> # Make predictions
    >>> probs = predictor.predict('villager', X_new)
    >>> labels = predictor.predict_label('seer', X_new)
"""

import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import itertools
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional, Dict, List, Any, Union
import warnings

from .data_preparation import prepare_data_for_training_with_meta

warnings.filterwarnings('ignore')


class RolePredictor:
    def __init__(
        self,
        csv_path_or_paths: Union[str, List[str]],
        lang_feature: bool = False,
        prep_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RolePredictor with data preparation.
        
        Parameters
        ----------
        csv_path_or_paths : str or List[str]
            Path(s) to CSV file(s) containing training data
        lang_feature : bool, default=False
            Whether to use language features (for future extension)
        prep_options : dict, optional
            Extra options passed to prepare_data_for_training_with_meta
            
        Raises
        ------
        ValueError
            If data preparation fails
        """
        print("="*70)
        print("INITIALIZING ROLE PREDICTOR")
        print("="*70)
        
        # Load and prepare data
        prep_options = prep_options or {}
        data_result = prepare_data_for_training_with_meta(
            csv_path_or_paths,
            lang_feature=lang_feature,
            **prep_options,
        )
        if data_result is None:
            raise ValueError("Failed to prepare data from CSV")
        
        # Unpack prepared data
        (self.X_train, self.X_test, self.y_train, self.y_test, 
         self.meta_train, self.meta_test, self.label_encoder, 
         self.feature_names, _, _) = data_result
        
        # Initialize model storage
        self.models = {}
        
        # Standard role counts for Werewolf game (5 players)
        self.role_counts = {'POSSESSED': 1, 'SEER': 1, 'VILLAGER': 2, 'WEREWOLF': 1}
        
        # Print initialization summary
        print(f"✓ Data prepared:")
        print(f"  - Training: {self.X_train.shape}")
        print(f"  - Test: {self.X_test.shape}")
        print(f"✓ Roles: {list(self.label_encoder.classes_)}")
        print(f"✓ Features: {len(self.feature_names)}")
        print("="*70 + "\n")
    
    # ======================================================================
    # Role Assignment Methods
    # ======================================================================
    
    def assign_roles_for_non_seer(
        self, logits: np.ndarray, y_batch: np.ndarray, fixed_self_role_name: str, 
        exec_id_batch: np.ndarray, attack_id_batch: np.ndarray, 
        day2_flag: bool = True, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        role_names = list(self.label_encoder.classes_)
        num_players = logits.shape[0]
        if num_players != 5:
            return np.array([]), np.array([])
        
        pred_list, y_list = [], []

        for self_index in range(num_players):
            self_role = y_batch[self_index]
            self_role_name = self.label_encoder.inverse_transform([self_role])[0]
            
            # Only process if actual role matches perspective
            if self_role_name != fixed_self_role_name:
                continue

            # Skip if self was executed or attacked (day 2+)
            if day2_flag and (self_index + 1 in exec_id_batch or self_index + 1 in attack_id_batch):
                if debug:
                    print(f"Skipping player {self_index + 1} (executed or attacked)")
                continue

            # Get other players' information
            other_indices = [i for i in range(num_players) if i != self_index]
            logits_others = logits[other_indices]
            y_others = y_batch[other_indices]
            
            # Collect constraints
            non_werewolf_indices_in_others = []
            if day2_flag:
                exec_id = exec_id_batch[self_index]
                attack_id = attack_id_batch[self_index]
                if np.isnan(attack_id) or np.isnan(exec_id): 
                    continue
                    
                attack_id_idx = int(attack_id) - 1
                exec_id_idx = int(exec_id) - 1
                if self_index == exec_id_idx or self_index == attack_id_idx: 
                    continue
                    
                if exec_id_idx in other_indices:
                    non_werewolf_indices_in_others.append(other_indices.index(exec_id_idx))
                if attack_id_idx in other_indices:
                    non_werewolf_indices_in_others.append(other_indices.index(attack_id_idx))
                non_werewolf_indices_in_others = list(set(non_werewolf_indices_in_others))
                
            # Create role pool for 4 other players
            reduced_counts = self.role_counts.copy()
            if reduced_counts.get(fixed_self_role_name, 0) == 0:
                continue
            reduced_counts[fixed_self_role_name] -= 1
            roles_pool = [role for role, count in reduced_counts.items() for _ in range(count)]
            if len(roles_pool) != 4: 
                continue
            
            # Find optimal role assignment
            best_perm = None
            best_score = -np.inf
            for perm in set(itertools.permutations(roles_pool)):
                # Check werewolf constraint
                if any(perm[idx] == "WEREWOLF" for idx in non_werewolf_indices_in_others):
                    continue
                
                # Calculate log-likelihood score
                score = sum(
                    np.log(logits_others[i][role_names.index(role)] + 1e-9) 
                    for i, role in enumerate(perm)
                )
                if score > best_score:
                    best_score, best_perm = score, perm
                    
            # Store result if valid assignment found
            if best_perm:
                pred_encoded = self.label_encoder.transform(list(best_perm))
                pred_list.append(pred_encoded)
                y_list.append(y_others)

        return (np.concatenate(pred_list) if pred_list else np.array([]),
                np.concatenate(y_list) if y_list else np.array([]))

    
    def assign_roles_for_seer_by_divination(
        self, logits: np.ndarray, y_batch: np.ndarray, 
        div_result_array1: np.ndarray, div_id_array1: np.ndarray, 
        div_result_array2: Optional[np.ndarray] = None, 
        div_id_array2: Optional[np.ndarray] = None,
        exec_id_batch: Optional[np.ndarray] = None, 
        attack_id_batch: Optional[np.ndarray] = None, 
        day2_flag: bool = True, debug: bool = False) -> dict:

        role_names = list(self.label_encoder.classes_)
        
        num_players = logits.shape[0]
        if num_players != 5:
            empty_result = (np.array([]), np.array([]))
            return {"black": empty_result, "white": empty_result}

        pred_list_black, y_list_black = [], []
        pred_list_white, y_list_white = [], []

        fixed_self_role_name = "SEER"
        werewolf_label = "WEREWOLF"

        for self_index in range(num_players):
            self_role_name = self.label_encoder.inverse_transform([y_batch[self_index]])[0]

            # Only process if actual role is SEER
            if self_role_name != fixed_self_role_name:
                continue
            
            # Skip if seer was executed or attacked (day 2+)
            if day2_flag and (self_index + 1 in exec_id_batch or self_index + 1 in attack_id_batch):
                continue

            if div_id_array1 is None or all(np.isnan(div_id_array1)):
                continue

            # Get other players' information
            other_indices = [i for i in range(num_players) if i != self_index]
            logits_others = logits[other_indices]
            y_others = y_batch[other_indices]

            # Collect constraints
            non_werewolf_indices_in_others = []
            div_constraints = {}  # {player_index: 'WEREWOLF'|'HUMAN'}

            # Execution/attack constraint
            if day2_flag and exec_id_batch is not None and attack_id_batch is not None:
                exec_id = exec_id_batch[self_index]
                attack_id = attack_id_batch[self_index]
                if not (np.isnan(exec_id) or np.isnan(attack_id)):
                    exec_id_idx = int(exec_id) - 1
                    attack_id_idx = int(attack_id) - 1
                else:
                    continue
                if exec_id_idx in other_indices:
                    non_werewolf_indices_in_others.append(other_indices.index(exec_id_idx))
                if attack_id_idx in other_indices:
                    non_werewolf_indices_in_others.append(other_indices.index(attack_id_idx))
                non_werewolf_indices_in_others = list(set(non_werewolf_indices_in_others))
            
            # Divination result constraints
            all_div_results = [(div_result_array1, div_id_array1)]
            if day2_flag and div_result_array2 is not None and div_id_array2 is not None:
                all_div_results.append((div_result_array2, div_id_array2))

            first_div_result_for_classification = None

            for i, (div_result_array, div_id_array) in enumerate(all_div_results):
                if np.isnan(div_id_array[self_index]): 
                    continue
                
                div_id = int(div_id_array[self_index]) - 1
                if div_id not in other_indices: 
                    continue
                    
                div_target_index_in_others = other_indices.index(div_id)
                div_result_str = str(div_result_array[self_index]).strip()

                # Parse divination result
                div_logic_result = None
                if div_result_str in ["WEREWOLF", "人狼", "黒"]:
                    div_logic_result = "WEREWOLF"
                elif div_result_str in ["HUMAN", "白", "not(人狼)"]:
                    div_logic_result = "HUMAN"
                
                if div_logic_result:
                    div_constraints[div_target_index_in_others] = div_logic_result
                    if i == 0:  # Save day 1 result for classification
                        first_div_result_for_classification = div_logic_result
            
            # Skip if no day 1 divination result
            if first_div_result_for_classification is None:
                continue

            # Create role pool
            reduced_counts = self.role_counts.copy()
            if reduced_counts.get(fixed_self_role_name, 0) == 0: 
                continue
            reduced_counts[fixed_self_role_name] -= 1
            roles_pool = [role for role, count in reduced_counts.items() for _ in range(count)]
            if len(roles_pool) != 4: 
                continue

            # Find optimal role assignment satisfying all constraints
            best_score, best_perm = -np.inf, None

            for perm in set(itertools.permutations(roles_pool)):
                is_valid = True
                
                # Check execution/attack constraint
                if any(perm[idx] == werewolf_label for idx in non_werewolf_indices_in_others):
                    continue
                    
                # Check divination constraints
                for target_idx, result in div_constraints.items():
                    if result == "WEREWOLF" and perm[target_idx] != werewolf_label:
                        is_valid = False
                        break
                    if result == "HUMAN" and perm[target_idx] == werewolf_label:
                        is_valid = False
                        break
                if not is_valid: 
                    continue

                # Calculate score
                score = sum(
                    np.log(logits_others[i][role_names.index(role)] + 1e-9) 
                    for i, role in enumerate(perm)
                )
                if score > best_score:
                    best_score, best_perm = score, perm

            # Store result classified by day 1 divination
            if best_perm:
                pred_encoded = self.label_encoder.transform(list(best_perm))
                if first_div_result_for_classification == "WEREWOLF":
                    pred_list_black.append(pred_encoded)
                    y_list_black.append(y_others)
                elif first_div_result_for_classification == "HUMAN":
                    pred_list_white.append(pred_encoded)
                    y_list_white.append(y_others)

        result = {
            "black": (np.concatenate(pred_list_black) if pred_list_black else np.array([]),
                      np.concatenate(y_list_black) if y_list_black else np.array([])),
            "white": (np.concatenate(pred_list_white) if pred_list_white else np.array([]),
                      np.concatenate(y_list_white) if y_list_white else np.array([])),
        }
        return result

    # ======================================================================
    # Training Methods
    # ======================================================================

    def _target_role_name_for_model(self, model_type: str) -> str:
        """Return target role name for F1 evaluation by perspective."""
        return 'POSSESSED' if model_type == 'werewolf' else 'WEREWOLF'


    def _target_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, model_type: str) -> float:
        """Compute single-target F1 (no macro averaging) for the requested perspective."""
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        target_label = self.label_encoder.transform([self._target_role_name_for_model(model_type)])[0]
        target_f1 = f1_score(y_true, y_pred, labels=[target_label], average=None, zero_division=0)
        return float(target_f1[0]) if target_f1.size > 0 else 0.0


    def _collect_constrained_assignments(
        self,
        model_type: str,
        preds_proba: np.ndarray,
        day2_flag: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect constrained assignment predictions/truths over all test games."""
        all_perspective_preds_list = []
        all_perspective_truths_list = []
        num_games = len(self.y_test) // 5
        num_players = 5

        for i in range(num_games):
            start_idx = i * num_players
            end_idx = (i + 1) * num_players
            game_logits = preds_proba[start_idx:end_idx]
            game_y_batch = self.y_test[start_idx:end_idx]
            game_meta = {
                'div_result1': self.meta_test['div_result1'][start_idx:end_idx],
                'div_id1': self.meta_test['div_id1'][start_idx:end_idx],
                'div_result2': self.meta_test['div_result2'][start_idx:end_idx],
                'div_id2': self.meta_test['div_id2'][start_idx:end_idx],
                'exec_id': self.meta_test['exec_id'][start_idx:end_idx],
                'attack_id': self.meta_test['attack_id'][start_idx:end_idx],
            }

            if model_type in {"villager", "possessed"}:
                fixed_role_name = "VILLAGER" if model_type == "villager" else "POSSESSED"
                preds_flat, truths_flat = self.assign_roles_for_non_seer(
                    logits=game_logits,
                    y_batch=game_y_batch,
                    fixed_self_role_name=fixed_role_name,
                    exec_id_batch=game_meta['exec_id'],
                    attack_id_batch=game_meta['attack_id'],
                    day2_flag=day2_flag,
                )
                if preds_flat.size > 0:
                    all_perspective_preds_list.append(preds_flat)
                    all_perspective_truths_list.append(truths_flat)

            elif model_type == "seer":
                seer_results_dict = self.assign_roles_for_seer_by_divination(
                    logits=game_logits,
                    y_batch=game_y_batch,
                    div_result_array1=game_meta['div_result1'],
                    div_id_array1=game_meta['div_id1'],
                    div_result_array2=game_meta['div_result2'],
                    div_id_array2=game_meta['div_id2'],
                    exec_id_batch=game_meta['exec_id'],
                    attack_id_batch=game_meta['attack_id'],
                    day2_flag=day2_flag,
                )
                for div_result_key in ["black", "white"]:
                    preds_flat, truths_flat = seer_results_dict[div_result_key]
                    if preds_flat.size > 0:
                        all_perspective_preds_list.append(preds_flat)
                        all_perspective_truths_list.append(truths_flat)

            elif model_type == "werewolf":
                for role_name in ["WEREWOLF", "POSSESSED", "VILLAGER"]:
                    preds_flat, truths_flat = self.assign_roles_for_non_seer(
                        logits=game_logits,
                        y_batch=game_y_batch,
                        fixed_self_role_name=role_name,
                        exec_id_batch=game_meta['exec_id'],
                        attack_id_batch=game_meta['attack_id'],
                        day2_flag=day2_flag,
                    )
                    if preds_flat.size > 0:
                        all_perspective_preds_list.append(preds_flat)
                        all_perspective_truths_list.append(truths_flat)

        if not all_perspective_truths_list:
            return np.array([]), np.array([])

        final_preds = np.concatenate(all_perspective_preds_list)
        final_truths = np.concatenate(all_perspective_truths_list)
        return final_preds, final_truths


    def evaluate_constrained_assignments(self, day2_flag: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run constrained final evaluation for all trained models on test split."""
        results: Dict[str, Dict[str, Any]] = {}
        for model_name in ["villager", "possessed", "seer", "werewolf"]:
            if model_name not in self.models:
                continue
            probs = self.models[model_name].predict_proba(self.X_test)
            final_preds, final_truths = self._collect_constrained_assignments(
                model_type=model_name,
                preds_proba=probs,
                day2_flag=day2_flag,
            )
            target_role_name = self._target_role_name_for_model(model_name)
            target_f1 = self._target_f1_score(final_truths, final_preds, model_name)
            results[model_name] = {
                "target_role": target_role_name,
                "target_f1": target_f1,
                "n_eval_samples": int(final_truths.size),
                "day2_flag": bool(day2_flag),
            }
        return results
    
    def _train_single_model(self, model_type: str = "villager", n_trials: int = 200, 
                            output_model_name: str = "model") -> xgb.XGBClassifier:
        
        def objective(trial):
            """Optuna objective function for hyperparameter optimization."""
            # Suggest hyperparameters
            params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'booster': 'gbtree',
                'enable_categorical': True, 
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'gamma': trial.suggest_float('gamma', 0.01, 5.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'seed': 42,
                'tree_method': 'hist',
                'early_stopping_rounds': 50,
            }
            
            # Train model
            model = xgb.XGBClassifier(**params)
            class_weights_train = compute_class_weight(
                class_weight='balanced', 
                classes=np.unique(self.y_train), 
                y=self.y_train
            )
            weights_train = [class_weights_train[label] for label in self.y_train]

            model.fit(self.X_train, self.y_train, sample_weight=weights_train,
                      eval_set=[(self.X_test, self.y_test)],
                      verbose=False)
            
            # Evaluate on test set
            preds_proba = model.predict_proba(self.X_test)
            final_preds, final_truths = self._collect_constrained_assignments(
                model_type=model_type,
                preds_proba=preds_proba,
                day2_flag=False,
            )

            if final_preds.size == 0 or final_truths.size == 0:
                return 0.0
            return self._target_f1_score(final_truths, final_preds, model_type)

        # ====== Optuna Optimization ======
        print(f"\n--- Training {model_type.upper()} Perspective Model ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best parameters: {study.best_params}")
        print(f"Best F1-score: {study.best_value:.4f}")

        # ====== Final Model Training ======
        best_params = study.best_params
        class_weights_train = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        weights_train = [class_weights_train[label] for label in self.y_train]

        final_model = xgb.XGBClassifier(
            objective='multi:softprob',
            use_label_encoder=False,
            seed=42,
            enable_categorical=True,
            **best_params,
            tree_method='hist',
        )
        
        print(f"Training final {model_type} model...")
        final_model.fit(self.X_train, self.y_train, sample_weight=weights_train)
        
        # Save model
        model_file = f"{output_model_name}_model.joblib"
        joblib.dump(final_model, model_file)
        print(f"✓ Model saved to {model_file}")
        
        return final_model

    
    def train(self, n_trials: int = 200) -> None:
        """
        Train all perspective models.
        
        Parameters
        ----------
        n_trials : int, default=200
            Number of Optuna trials for each model
        """
        print("\n" + "="*70)
        print("TRAINING ALL PERSPECTIVE MODELS")
        print("="*70)
        
        # Train villager perspective model
        self.models['villager'] = self._train_single_model(
            model_type="villager",
            n_trials=n_trials,
            output_model_name="XGB_villager_perspective"
        )

        # Train possessed perspective model
        self.models['possessed'] = self._train_single_model(
            model_type="possessed",
            n_trials=n_trials,
            output_model_name="XGB_possessed_perspective"
        )
        
        # Train seer perspective model
        self.models['seer'] = self._train_single_model(
            model_type="seer",
            n_trials=n_trials,
            output_model_name="XGB_seer_perspective"
        )
        
        # Train werewolf perspective model
        self.models['werewolf'] = self._train_single_model(
            model_type="werewolf",
            n_trials=n_trials,
            output_model_name="XGB_werewolf_perspective"
        )
        
        print("\n" + "="*70)
        print("✓ ALL MODELS TRAINED SUCCESSFULLY")
        print("="*70)

    # ======================================================================
    # Prediction Methods
    # ======================================================================
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using a specific model.
        
        Parameters
        ----------
        model_name : str
            Model perspective: "villager", "possessed", "seer", or "werewolf"
        X : np.ndarray
            Input features (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted role probabilities (n_samples, n_roles)
            
        Raises
        ------
        ValueError
            If model not found or feature mismatch
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.models.keys())}"
            )
        
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Feature mismatch: expected {self.X_train.shape[1]}, got {X.shape[1]}"
            )
        
        model = self.models[model_name]
        return model.predict_proba(X)
    
    
    def predict_label(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Predict role labels for input features.
        
        Parameters
        ----------
        model_name : str
            Model perspective: "villager", "possessed", "seer", or "werewolf"
        X : np.ndarray
            Input features (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted encoded role labels
            
        Raises
        ------
        ValueError
            If model not found or feature mismatch
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.models.keys())}"
            )
        
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Feature mismatch: expected {self.X_train.shape[1]}, got {X.shape[1]}"
            )
        
        model = self.models[model_name]
        return model.predict(X)
    
    
    def predict_role_names(self, model_name: str, X: np.ndarray) -> List[str]:
        """
        Predict role names (decoded labels) for input features.
        
        Parameters
        ----------
        model_name : str
            Model perspective: "villager", "possessed", "seer", or "werewolf"
        X : np.ndarray
            Input features (n_samples, n_features)
            
        Returns
        -------
        List[str]
            Predicted role names
        """
        labels = self.predict_label(model_name, X)
        return [self.label_encoder.inverse_transform([label])[0] for label in labels]
    
    
    def load_model(self, model_name: str, model_path: str) -> None:
        """
        Load a pre-trained model from file.
        
        Parameters
        ----------
        model_name : str
            Name to assign to loaded model: "villager", "possessed", "seer", or "werewolf"
        model_path : str
            Path to saved model file (.joblib)
        """
        self.models[model_name] = joblib.load(model_path)
        print(f"✓ Loaded model '{model_name}' from {model_path}")
    
    
    def save_model(self, model_name: str, model_path: str) -> None:
        """
        Save a trained model to file.
        
        Parameters
        ----------
        model_name : str
            Name of model to save: "villager", "possessed", "seer", or "werewolf"
        model_path : str
            Path where model will be saved (.joblib)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        joblib.dump(self.models[model_name], model_path)
        print(f"✓ Saved model '{model_name}' to {model_path}")


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    """
    Example: Initialize predictor, train models, and make predictions
    """
    csv_path = r"../../../all_feature_table_2025sp17_with_talks.csv"
    
    # Initialize predictor
    print("\n" + "="*70)
    print("EXAMPLE USAGE")
    print("="*70)
    
    predictor = RolePredictor(csv_path, lang_feature=True)
    
    # Train all models
    predictor.train(n_trials=200)
    
    # Make predictions on test set
    print("\n" + "="*70)
    print("MAKING PREDICTIONS")
    print("="*70)
    
    # Predict probabilities using villager perspective
    probs = predictor.predict("villager", predictor.X_test[:5])
    print(f"\nVillager perspective probabilities (first 5 samples):")
    print(f"Shape: {probs.shape}")
    
    # Predict labels using seer perspective
    labels = predictor.predict_label("seer", predictor.X_test[:5])
    role_names = predictor.predict_role_names("seer", predictor.X_test[:5])
    print(f"\nSeer perspective predictions (first 5 samples):")
    print(f"Encoded labels: {labels}")
    print(f"Role names: {role_names}")
    
    # Verify training completion
    print(f"\n✓ Training complete! Available models: {list(predictor.models.keys())}")
