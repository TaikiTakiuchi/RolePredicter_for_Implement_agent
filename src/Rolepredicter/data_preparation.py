"""
Data Preparation Functions

This module provides functions to load, preprocess, and prepare data for role prediction model training.
Handles data loading from CSV, feature engineering, train-test splitting, and metadata extraction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from typing import Tuple, Optional, Dict, List, Any, Union


DEFAULT_LEAKAGE_COLUMNS = [
    'True_Div_recepient_id_1',
    'True_Div_result_1',
    'True_Div_recepient_id_2',
    'True_Div_result_2',
    'target_total_votes',
]


def _ensure_csv_paths(csv_path_or_paths: Union[str, List[str]]) -> List[str]:
    """Normalize CSV path input into a non-empty list."""
    if isinstance(csv_path_or_paths, str):
        return [csv_path_or_paths]
    if isinstance(csv_path_or_paths, list) and csv_path_or_paths:
        return csv_path_or_paths
    raise ValueError("csv_path_or_paths must be a non-empty string or list of strings")


def _load_and_concat_csv(csv_paths: List[str]) -> pd.DataFrame:
    """Load and concatenate multiple CSV files with a source marker column."""
    dataframes = []
    for csv_path in csv_paths:
        df_part = pd.read_csv(csv_path)
        df_part['dataset_source'] = csv_path
        dataframes.append(df_part)

    if not dataframes:
        raise ValueError("No CSV files were loaded")

    return pd.concat(dataframes, axis=0, ignore_index=True)


def prepare_data_for_training_with_meta(
    csv_path_or_paths: Union[str, List[str]],
    lang_feature: bool = False,
    day_filter: int = 1,
    leakage_drop_columns: Optional[List[str]] = None,
    group_column: str = 'source_file',
    test_size: float = 0.2,
    split_mode: str = 'group_shuffle',
    n_splits: int = 5,
    fold_index: int = 0,
) -> Optional[Tuple]:
    """
    CSVファイルからデータを読み込み、訓練用・テスト用データを準備する。
    
    処理内容：
    1. CSVからデータロード（day=1 のみ）
    2. 目的変数（役職）のエンコード
    3. GroupShuffleSplit で訓練・テストデータ分割
    4. カテゴリカル特徴量のエンコード
    5. メタデータ（占い結果、追放者IDなど）の抽出
    6. 数値特徴量とカテゴリカル特徴量の分離・結合
    
    Parameters
    ----------
    csv_path_or_paths : str or List[str]
        読み込むCSVファイルのパスまたはパス一覧
    lang_feature : bool, default=False
        後方互換のための引数（現在は未使用）
        
    Returns
    -------
    tuple or None
        成功時：(X_train, X_test, y_train, y_test, meta_train, meta_test, 
                  role_encoder, training_columns, scaler, feature_encoder)
        失敗時：None
        
    Raises
    ------
    None
        エラー時は None を返す（詳細はコンソール出力を参照）
    """
    print("--- 1. Starting Data Processing ---")
    csv_paths = _ensure_csv_paths(csv_path_or_paths)
    leakage_drop_columns = leakage_drop_columns or DEFAULT_LEAKAGE_COLUMNS
    
    # データの読み込みと基本フィルタリング
    df = _load_and_concat_csv(csv_paths)
    df = df[df["day"] == day_filter].copy()
    print(f"Successfully loaded {len(csv_paths)} CSV file(s) and filtered for day {day_filter}.")
    print(f"Loaded datasets: {', '.join(csv_paths)}")
    print(f"Data shape: {df.shape}")

    # 目的変数 'role' のエンコード
    role_encoder = LabelEncoder()
    try:
        df['role'] = role_encoder.fit_transform(df['role'].astype(str))
        print(f"Encoded target column 'role'. Classes: {role_encoder.classes_}")
    except Exception as e:
        print(f"Error: Could not encode target column 'role'. Error: {e}")
        return None
    
    # GroupShuffleSplit で訓練・テストデータを分割
    y_full = df['role'].values
    if group_column not in df.columns:
        print(f"Warning: group column '{group_column}' not found. Falling back to 'source_file'.")
        group_column = 'source_file'
    groups = df[group_column].values

    try:
        if split_mode == 'group_kfold':
            if n_splits < 2:
                raise ValueError("n_splits must be >= 2 when split_mode='group_kfold'")
            gkf = GroupKFold(n_splits=n_splits)
            all_splits = list(gkf.split(df, y_full, groups=groups))
            if fold_index < 0 or fold_index >= len(all_splits):
                raise ValueError(
                    f"fold_index out of range: {fold_index}. Must be 0..{len(all_splits)-1}"
                )
            train_idx, test_idx = all_splits[fold_index]
            print(f"Using GroupKFold split: fold {fold_index+1}/{len(all_splits)}")
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_idx, test_idx = next(gss.split(df, y_full, groups=groups))
            print(f"Using GroupShuffleSplit: test_size={test_size}")
    except Exception as e:
        print(f"Error during data split. Error: {e}")
        return None
        
    df_train, df_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y_train, y_test = y_full[train_idx], y_full[test_idx]
    print(f"Data split: Train={len(df_train)}, Test={len(df_test)}")

    # カテゴリカル特徴量のエンコード
    result_like_cols = [c for c in df.columns if c.endswith('_result')]
    role_like_cols = [c for c in df.columns if c.endswith('_role') or c.endswith('_roles')]
    categorical_features_list = result_like_cols + role_like_cols
    categorical_features = [col for col in categorical_features_list if col in df_train.columns]
    
    # 'Div_result' 列の特別処理
    df_train['div_result_error'] = df_train['Div_result'].apply(lambda x: x == "error").astype(int)
    df_test['div_result_error'] = df_test['Div_result'].apply(lambda x: x == "error").astype(int)
    div_map = {'白': -1, '黒': 1}
    
    encoder = None  # 後で使用するため保存
    for col in categorical_features:
        if 'Div_result' in col:
            df_train[col] = df_train[col].map(div_map).fillna(0).astype(int)
            df_test[col] = df_test[col].map(div_map).fillna(0).astype(int)
        else:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df_train[col] = encoder.fit_transform(df_train[[col]])
            df_test[col] = encoder.transform(df_test[[col]])

    # NOTE: speaker prediction model and speaker-probability-derived features were removed.
    if lang_feature:
        print("Info: 'lang_feature' is currently ignored. Speaker probability features are disabled.")

    # メタデータの抽出
    DiV_cols = {
        'div_result1': 'True_Div_result_1', 
        'div_id1': 'True_Div_recepient_id_1',
        'div_result2': 'True_Div_result_2', 
        'div_id2': 'True_Div_recepient_id_2'
    }
    Dead_cols = {'exec_id': 'exec_id', 'attack_id': 'attack_id'}
    meta_cols = {**DiV_cols, **Dead_cols}

    def extract_meta(df_part: pd.DataFrame, meta_cols_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
        """メタデータをデータフレームから抽出"""
        meta_data = {}
        for key, col_name in meta_cols_dict.items():
            if col_name in df_part.columns:
                meta_data[key] = df_part[col_name].values
            else:
                print(f"Warning: Metadata column '{col_name}' not found. Filling with NaN.")
                meta_data[key] = np.full(len(df_part), np.nan)
        return meta_data

    meta_train = extract_meta(df_train, meta_cols)
    meta_test = extract_meta(df_test, meta_cols)
    print("Extracted metadata for train and test sets.")

    # 特徴量カラムの定義と除外
    base_drop_cols = [
        'id', 'role', 'source_file', 'day', 'role_encoded', 
        'Est_id_Fact_role', 'Est_id_Est_roles', 'character_name', 'agent_name', 
        'combined_text', 'seer_co_order', 'alive', 'dataset_source'
    ]
    vote_cols = [c for c in df.columns if 'vote_id' in c]
    id_like_cols = [c for c in df.columns if c.endswith('_id')]
    flag_like_cols = [c for c in df.columns if c.endswith('_flag')]
    
    all_drop_cols = list(set(
        base_drop_cols + 
        leakage_drop_columns +
        list(meta_cols.values()) + 
        id_like_cols + 
        flag_like_cols +
        vote_cols
    ))
    
    X_train_raw = df_train.drop(columns=all_drop_cols, errors='ignore')
    X_test_raw = df_test.drop(columns=all_drop_cols, errors='ignore')

    # 数値特徴量とカテゴリカル特徴量の分離
    actual_cat_features = [col for col in categorical_features if col in X_train_raw.columns]
    numeric_features = [col for col in X_train_raw.columns if col not in actual_cat_features]
    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:5]}...")
    print(f"Categorical features ({len(actual_cat_features)}): {actual_cat_features}")
    
    # テストデータのカラムを訓練データに合わせる
    training_columns = X_train_raw.columns.tolist()
    X_test_raw = X_test_raw.reindex(columns=training_columns, fill_value=0) 
    
    # 訓練データの準備
    X_train_num_df = X_train_raw[numeric_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_train_scaled = X_train_num_df.values
    X_train_cat = X_train_raw[actual_cat_features].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # テストデータの準備
    X_test_num_df = X_test_raw[numeric_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_scaled = X_test_num_df.values
    X_test_cat = X_test_raw[actual_cat_features].apply(pd.to_numeric, errors='coerce').fillna(0).values

    # 次元を揃える
    if X_train_cat.ndim == 1: 
        X_train_cat = X_train_cat.reshape(-1, 1)
    if X_test_cat.ndim == 1: 
        X_test_cat = X_test_cat.reshape(-1, 1)
        
    # 数値特徴量とカテゴリカル特徴量を結合
    X_train = np.hstack([X_train_scaled, X_train_cat])
    X_test = np.hstack([X_test_scaled, X_test_cat])
    
    final_training_columns = numeric_features + actual_cat_features

    # Safety check: leaked columns must not remain in model inputs.
    leaked_in_final = [c for c in leakage_drop_columns if c in final_training_columns]
    if leaked_in_final:
        print(f"Error: leakage columns still in feature set: {leaked_in_final}")
        return None
    
    print(f"Final feature shape: X_train={X_train.shape}, X_test={X_test.shape}")
    print("--- Data Processing Finished ---\n")

    return (
        X_train, X_test, y_train, y_test, 
        meta_train, meta_test, 
        role_encoder, 
        final_training_columns, 
        None,
        encoder 
    )
