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
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# プロジェクトのsrcディレクトリをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Rolepredictorモジュールのインポート
from Rolepredicter.role_predictor import RolePredictor
from Rolepredicter.data_preparation import prepare_data_for_training_with_meta


def get_data_path() -> str:
    """
    データファイルのパスを相対パスで取得
    
    Returns
    -------
    str
        CSVファイルのフルパス
    """
    pipeline_dir = Path(__file__).parent
    data_dir = pipeline_dir.parent.parent / "data"
    
    # dataディレクトリ内で最新のCSVファイルを探す
    csv_files = list(data_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # ファイル名でソート（最新のものを選択）
    latest_csv = sorted(csv_files)[-1]
    print(f"✓ Data file selected: {latest_csv.name}")
    return str(latest_csv)


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


def display_prediction_results(predictor: RolePredictor) -> None:
    """
    予測結果を表示
    
    Parameters
    ----------
    predictor : RolePredictor
        学習済みのRolePredicterインスタンス
    """
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    # テストデータの最初の10サンプルで予測
    n_samples = min(10, predictor.X_test.shape[0])
    X_sample = predictor.X_test[:n_samples]
    y_sample = predictor.y_test[:n_samples]
    
    for model_name in ["human", "seer", "werewolf"]:
        print(f"\n--- {model_name.upper()} Perspective Model ---")
        
        # 予測
        predictions = predictor.predict_role_names(model_name, X_sample)
        true_roles = [predictor.label_encoder.inverse_transform([y])[0] for y in y_sample]
        
        # 結果を表示
        print(f"\nSample Predictions (first {n_samples}):")
        print(f"{'Sample':<8} {'True Role':<15} {'Predicted Role':<15} {'Match':<8}")
        print("-" * 50)
        
        matches = 0
        for i, (true_role, pred_role) in enumerate(zip(true_roles, predictions)):
            is_match = "✓" if true_role == pred_role else "✗"
            if true_role == pred_role:
                matches += 1
            print(f"{i+1:<8} {true_role:<15} {pred_role:<15} {is_match:<8}")
        
        accuracy = (matches / n_samples) * 100
        print(f"\nAccuracy (first {n_samples}): {accuracy:.1f}%")


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
    
    for model_name in ["human", "seer", "werewolf"]:
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
    
    for model_name in ["human", "seer", "werewolf"]:
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
        
        data_path = get_data_path()
        models_dir = get_models_dir()
        
        print(f"✓ Models directory: {models_dir.relative_to(Path.cwd()) if Path.cwd() in models_dir.parents else models_dir}")
        
        # RolePredicterの初期化
        predictor = RolePredictor(data_path, lang_feature=True)
        
        # データ情報の表示
        display_data_info(predictor)
        
        # ========== ステップ2: モデル学習 ==========
        print("\n" + "="*70)
        print("STEP 2: Model Training")
        print("="*70)
        
        # n_trialsは本番環境では200-500を推奨（デモ用に50に設定）
        n_trials = 50  # 開発用：50 本番用：200-500
        print(f"\nStarting hyperparameter optimization with {n_trials} trials per model...\n")
        
        predictor.train(n_trials=n_trials)
        
        # ========== ステップ3: 予測結果の表示 ==========
        display_prediction_results(predictor)
        
        # ========== ステップ4: 特徴量重要度の表示 ==========
        display_feature_importance(predictor)
        
        # ========== ステップ5: モデルの保存 ==========
        save_models(predictor, models_dir)
        
        # ========== 完了表示 ==========
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nTraining Summary:")
        print(f"  - Data: {Path(data_path).name}")
        print(f"  - Models saved to: {models_dir}")
        print(f"  - Training samples: {predictor.X_train.shape[0]}")
        print(f"  - Test samples: {predictor.X_test.shape[0]}")
        print(f"  - Trained models: {list(predictor.models.keys())}")
        print(f"  - Optimization trials: {n_trials} per model")
        print("="*70 + "\n")
        
        return predictor
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    predictor = main()
