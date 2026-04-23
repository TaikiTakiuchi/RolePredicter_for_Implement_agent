"""
Training Pipeline for Werewolf Role Prediction

このパイプラインは以下のステップを実行します：
1. データの取得と加工
2. モデルの学習
3. 結果の評価と表示
4. モデルの保存

データは相対パス `../../data/` から、モデルは相対パス `../../models/` に保存されます。
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List

# プロジェクトのsrcディレクトリをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


# Rolepredictorモジュールのインポート
from Rolepredicter.role_predictor import RolePredictor
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "training_config.json"


def load_training_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load training configuration from JSON."""
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_data_paths(config: Dict[str, Any], data_type: str = "day1") -> List[str]:
    """
    データファイルのパスを相対パスで取得
    
    Returns
    -------
    str
        CSVファイルのフルパス
    """
    if data_type == "day2":
        configured_paths = config.get("data_paths_day2", [])
    else:
        configured_paths = config.get("data_paths", [])
    if not configured_paths:
        raise ValueError("No data_paths defined in training config")

    resolved_paths: List[str] = []
    for p in configured_paths:
        path_obj = Path(p)
        if not path_obj.is_absolute():
            path_obj = PROJECT_ROOT / path_obj
        if not path_obj.exists():
            raise FileNotFoundError(f"Configured data file not found: {path_obj}")
        resolved_paths.append(str(path_obj))

    print("✓ Data files selected:")
    for p in resolved_paths:
        print(f"  - {Path(p).name}")
    return resolved_paths

def get_data_paths_day2(config: Dict[str, Any]) -> List[str]:
    """
    データファイルのパスを相対パスで取得（Day2用）
    
    Returns
    -------
    str
        CSVファイルのフルパス
    """
    configured_paths = config.get("data_paths_day2", [])
    if not configured_paths:
        raise ValueError("No data_paths_day2 defined in training config")

    resolved_paths: List[str] = []
    for p in configured_paths:
        path_obj = Path(p)
        if not path_obj.is_absolute():
            path_obj = PROJECT_ROOT / path_obj
        if not path_obj.exists():
            raise FileNotFoundError(f"Configured data file not found: {path_obj}")
        resolved_paths.append(str(path_obj))

    print("✓ Day2 data files selected:")
    for p in resolved_paths:
        print(f"  - {Path(p).name}")
    return resolved_paths

def get_models_dir() -> Path:
    """
    モデル保存ディレクトリを相対パスで取得
    
    Returns
    -------
    Path
        モデル保存ディレクトリ
    """
    pipeline_dir = Path(__file__).parent
    models_dir = pipeline_dir.parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def display_data_info(predictor: RolePredictor) -> None:
    """
    データ情報を表示
    
    Parameters
    ----------
    predictor : RolePredictor
        初期化されたRolePredicterインスタンス
    """
    print("\n" + "="*70)
    print("DATA INFORMATION")
    print("="*70)
    
    print(f"\nTraining Data:")
    print(f"  Shape: {predictor.X_train.shape}")
    print(f"  Samples: {predictor.X_train.shape[0]}")
    print(f"  Features: {predictor.X_train.shape[1]}")
    
    print(f"\nTest Data:")
    print(f"  Shape: {predictor.X_test.shape}")
    print(f"  Samples: {predictor.X_test.shape[0]}")
    print(f"  Features: {predictor.X_test.shape[1]}")
    
    print(f"\nRoles:")
    roles = list(predictor.label_encoder.classes_)
    print(f"  Available: {roles}")
    
    # 訓練データの役職分布
    unique_train, counts_train = np.unique(predictor.y_train, return_counts=True)
    print(f"\nTraining Data Role Distribution:")
    for role_id, count in zip(unique_train, counts_train):
        role_name = predictor.label_encoder.inverse_transform([role_id])[0]
        percentage = (count / len(predictor.y_train)) * 100
        print(f"  {role_name}: {count} ({percentage:.1f}%)")
    
    # テストデータの役職分布
    unique_test, counts_test = np.unique(predictor.y_test, return_counts=True)
    print(f"\nTest Data Role Distribution:")
    for role_id, count in zip(unique_test, counts_test):
        role_name = predictor.label_encoder.inverse_transform([role_id])[0]
        percentage = (count / len(predictor.y_test)) * 100
        print(f"  {role_name}: {count} ({percentage:.1f}%)")


def display_target_class_f1_summary(score_df: pd.DataFrame) -> None:
    """Display target-class F1 summary table."""
    print("\n" + "="*70)
    print("TARGET-CLASS F1 SUMMARY")
    print("="*70)
    if score_df.empty:
        print("No score data available")
        return
    print(score_df.to_string(index=False))


def save_models(predictor: RolePredictor, models_dir: Path) -> None:
    """
    訓練済みモデルを保存
    
    Parameters
    ----------
    predictor : RolePredictor
        学習済みのRolePredicterインスタンス
    models_dir : Path
        モデル保存ディレクトリ
    """
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70 + "\n")
    
    for model_name in ["villager", "possessed", "seer", "werewolf"]:
        model_path = models_dir / f"{model_name}_model.joblib"
        predictor.save_model(model_name, str(model_path))


def display_feature_importance(predictor: RolePredictor) -> None:
    """
    特徴量の重要度を表示（XGBoostモデル用）
    
    Parameters
    ----------
    predictor : RolePredictor
        学習済みのRolePredicterインスタンス
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    
    for model_name in ["villager", "possessed", "seer", "werewolf"]:
        print(f"\n--- {model_name.upper()} Model ---")
        
        model = predictor.models[model_name]
        importance_dict = model.get_booster().get_score(importance_type='weight')
        
        if not importance_dict:
            print("No feature importance data available")
            continue
        
        # 重要度でソート（上位10個）
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"{'Feature':<25} {'Importance':<15} {'Percentage':<15}")
        print("-" * 55)
        
        total_importance = sum([v for _, v in sorted_importance])
        for feature, importance in sorted_importance:
            percentage = (importance / total_importance) * 100
            print(f"{feature:<25} {importance:<15} {percentage:>6.1f}%")


def save_feature_importance_report(predictor: RolePredictor, models_dir: Path) -> Path:
    """Save top feature importance for each perspective model as CSV."""
    rows = []
    for model_name in ["villager", "possessed", "seer", "werewolf"]:
        model = predictor.models.get(model_name)
        if model is None:
            continue
        importance_dict = model.get_booster().get_score(importance_type='weight')
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in sorted_importance) or 1.0
        for rank, (feature, importance) in enumerate(sorted_importance, start=1):
            rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "feature": feature,
                    "importance": importance,
                    "importance_ratio": importance / total,
                }
            )

    report_path = models_dir / "feature_importance_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"✓ Feature importance report saved: {report_path}")
    return report_path


def display_final_constrained_scores(predictor: RolePredictor) -> pd.DataFrame:
    """Display final constrained-evaluation scores by perspective."""
    print("\n" + "="*70)
    print("FINAL CONSTRAINED EVALUATION")
    print("="*70)
    print("F1 target rule: werewolf model -> POSSESSED, others -> WEREWOLF")

    results = predictor.evaluate_constrained_assignments(day2_flag=False)
    rows = []
    for model_name, result in results.items():
        rows.append(
            {
                "model": model_name,
                "target_role": result["target_role"],
                "target_f1": result["target_f1"],
                "n_eval_samples": result["n_eval_samples"],
                "day2_flag": result["day2_flag"],
            }
        )

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        print("No constrained-evaluation results available.")
        return score_df

    print(score_df.to_string(index=False))
    return score_df


def save_final_constrained_scores(score_df: pd.DataFrame, models_dir: Path) -> Path:
    """Save final constrained-evaluation scores to CSV."""
    output_path = models_dir / "final_constrained_scores.csv"
    score_df.to_csv(output_path, index=False)
    print(f"✓ Final constrained scores saved: {output_path}")
    return output_path


def save_training_metadata(
    models_dir: Path,
    config: Dict[str, Any],
    data_paths: List[str],
    predictor: RolePredictor,
) -> Path:
    """Save reproducibility metadata for this training run."""
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_paths": data_paths,
        "config": config,
        "feature_count": int(predictor.X_train.shape[1]),
        "feature_names": list(predictor.feature_names),
        "train_samples": int(predictor.X_train.shape[0]),
        "test_samples": int(predictor.X_test.shape[0]),
        "label_classes": [str(c) for c in predictor.label_encoder.classes_],
        "trained_models": list(predictor.models.keys()),
    }

    metadata_path = models_dir / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Training metadata saved: {metadata_path}")
    return metadata_path


def main():
    """
    パイプラインのメイン処理
    """
    print("\n" + "="*70)
    print("WEREWOLF ROLE PREDICTION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    try:
        # ========== ステップ1: データ取得と初期化 ==========
        print("STEP 1: Data Preparation")
        print("-" * 70)

        config = load_training_config()
        data_paths = get_data_paths(config)
        cv_folds = int(config.get("cv_folds", 1))
        prep_options = {
            "day_filter": int(config.get("day_filter", 1)),
            "leakage_drop_columns": config.get("leakage_drop_columns", []),
            "group_column": config.get("group_column", "source_file"),
            "test_size": float(config.get("test_size", 0.2)),
            "split_mode": "group_kfold" if cv_folds > 1 else "group_shuffle",
            "n_splits": cv_folds if cv_folds > 1 else 5,
            "fold_index": 0,
        }
        lang_feature = bool(config.get("lang_feature", True))
        models_dir = get_models_dir()
        
        print(f"✓ Models directory: {models_dir.relative_to(Path.cwd()) if Path.cwd() in models_dir.parents else models_dir}")
        
        n_trials = int(config.get("n_trials", 50))
        print("\n" + "="*70)
        print("STEP 2: Model Training")
        print("="*70)
        print(f"\nStarting hyperparameter optimization with {n_trials} trials per model...\n")

        cv_rows = []
        predictor = None

        if cv_folds > 1:
            for fold_idx in range(cv_folds):
                print("\n" + "-"*70)
                print(f"FOLD {fold_idx+1}/{cv_folds}")
                print("-"*70)
                fold_prep_options = dict(prep_options)
                fold_prep_options["fold_index"] = fold_idx

                fold_predictor = RolePredictor(data_paths, lang_feature=lang_feature, prep_options=fold_prep_options)
                display_data_info(fold_predictor)
                fold_predictor.train(n_trials=n_trials)
                fold_score_df = pd.DataFrame.from_dict(
                    fold_predictor.evaluate_constrained_assignments(day2_flag=False), orient='index'
                ).reset_index().rename(columns={'index': 'model'})
                fold_score_df["fold"] = fold_idx + 1
                cv_rows.append(fold_score_df)

                # 最後のfoldモデルを返り値用に保持
                predictor = fold_predictor

            cv_score_df = pd.concat(cv_rows, ignore_index=True) if cv_rows else pd.DataFrame()
            if not cv_score_df.empty:
                cv_score_df = cv_score_df[["fold", "model", "target_role", "target_f1", "n_eval_samples", "day2_flag"]]
                display_target_class_f1_summary(cv_score_df)
                cv_score_df.to_csv(models_dir / "cv_target_f1_scores.csv", index=False)
                print(f"✓ CV target F1 scores saved: {models_dir / 'cv_target_f1_scores.csv'}")

                mean_df = (
                    cv_score_df.groupby(["model", "target_role"], as_index=False)["target_f1"]
                    .mean()
                    .rename(columns={"target_f1": "mean_target_f1"})
                )
                print("\nSample Predictions (5-fold mean target-class F1):")
                print(mean_df.to_string(index=False))
                mean_df.to_csv(models_dir / "cv_target_f1_scores_mean.csv", index=False)
                print(f"✓ CV mean target F1 saved: {models_dir / 'cv_target_f1_scores_mean.csv'}")
        else:
            predictor = RolePredictor(data_paths, lang_feature=lang_feature, prep_options=prep_options)
            display_data_info(predictor)
            predictor.train(n_trials=n_trials)

            # ========== ステップ3: 特徴量重要度の表示 ==========
            display_feature_importance(predictor)
            save_feature_importance_report(predictor, models_dir)

            # ========== ステップ4: 最終制約付き評価 ==========
            final_score_df = display_final_constrained_scores(predictor)
            if not final_score_df.empty:
                save_final_constrained_scores(final_score_df, models_dir)
                display_target_class_f1_summary(final_score_df)

            # ========== ステップ5: モデルの保存 ==========
            save_models(predictor, models_dir)
            save_training_metadata(models_dir, config, data_paths, predictor)
        
        # ========== 完了表示 ==========
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nTraining Summary:")
        print(f"  - Data files: {len(data_paths)}")
        for p in data_paths:
            print(f"      * {Path(p).name}")
        print(f"  - Models saved to: {models_dir}")
        print(f"  - Training samples: {predictor.X_train.shape[0]}")
        print(f"  - Test samples: {predictor.X_test.shape[0]}")
        print(f"  - Trained models: {list(predictor.models.keys())}")
        print(f"  - Optimization trials: {n_trials} per model")
        print(f"  - CV folds: {cv_folds}")
        print("="*70 + "\n")
        
        return predictor
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    predictor = main()
