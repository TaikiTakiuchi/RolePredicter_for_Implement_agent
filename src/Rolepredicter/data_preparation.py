"""
Data Preparation Functions

This module provides functions to load, preprocess, and prepare data for role prediction model training.
Handles data loading from CSV, feature engineering, train-test splitting, and metadata extraction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier as XGB
from sklearn.metrics import accuracy_score
from typing import Tuple, Optional, Dict, List, Any


def get_speaker_probability_features(df_train_in: pd.DataFrame, df_test_in: pd.DataFrame, use_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    訓練データとテストデータから、各エージェントの予測確率と予測クラス名を
    元のデータフレームに列として追加して返す。
    
    このモデルでは、TF-IDF特徴量と指定された特徴量を組み合わせて、
    キャリブレーション付きXGBoostで各話者を予測します。
    
    Parameters
    ----------
    df_train_in : pd.DataFrame
        訓練用データフレーム
    df_test_in : pd.DataFrame
        テスト用データフレーム
    use_features : List[str]
        話者確率特徴量を計算する際に使用する特徴量のリスト
        例：["ReqDiscuss", "ReqListen", "Req(CO)", ...]
        
    Returns
    -------
    tuple
        (df_train, df_test) 話者確率特徴量が追加されたデータフレーム
        
    Notes
    -----
    追加される列：
    - prob_class_<agent_name>: 各エージェントの予測確率
    - predicted_class: 予測された話者名
    """
    df_train = df_train_in.copy()
    df_test = df_test_in.copy()

    # エージェント名のラベルエンコーディング
    agent_le = LabelEncoder()
    all_agents = pd.concat([df_train['agent_name'], df_test['agent_name']])
    agent_le.fit(all_agents)
    
    y_train = agent_le.transform(df_train['agent_name'])
    y_test = agent_le.transform(df_test['agent_name'])
    classes = agent_le.classes_

    # テキストのベクトル化 (TF-IDF)
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', 
        ngram_range=(2, 5), 
        max_features=2000,
        token_pattern=None
    )
    tfidf_train = vectorizer.fit_transform(df_train['combined_text'].astype(str)).toarray()
    tfidf_test = vectorizer.transform(df_test['combined_text'].astype(str)).toarray()

    # 特徴量の結合
    X_train = np.hstack([tfidf_train, df_train[use_features].fillna(0).values])
    X_test = np.hstack([tfidf_test, df_test[use_features].fillna(0).values])

    # モデル設定（XGBoost + Calibration）
    print("Training speaker prediction model...")
    params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 42}
    model = XGB(**params, eval_metric='mlogloss')
    calibrated = CalibratedClassifierCV(estimator=model, cv=5)

    # Trainの予測確率算出 (OOF)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_probs = cross_val_predict(
        calibrated, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1
    )

    # Testの予測確率算出
    calibrated.fit(X_train, y_train)
    test_probs = calibrated.predict_proba(X_test)

    # 精度表示
    train_accuracy = accuracy_score(y_train, np.argmax(train_probs, axis=1))
    test_accuracy = accuracy_score(y_test, np.argmax(test_probs, axis=1))
    print(f"Speaker Model Train Accuracy: {train_accuracy:.4f}")
    print(f"Speaker Model Test Accuracy: {test_accuracy:.4f}")

    # 確率列の追加
    for i, class_name in enumerate(classes):
        df_train[f'prob_class_{class_name}'] = train_probs[:, i]
        df_test[f'prob_class_{class_name}'] = test_probs[:, i]

    # 最も確率が高いクラス名の追加
    df_train['predicted_class'] = [classes[np.argmax(p)] for p in train_probs]
    df_test['predicted_class'] = [classes[np.argmax(p)] for p in test_probs]

    return df_train, df_test


def prepare_data_for_training_with_meta(csv_path: str, lang_feature: bool = False) -> Optional[Tuple]:
    """
    CSVファイルからデータを読み込み、訓練用・テスト用データを準備する。
    
    処理内容：
    1. CSVからデータロード（day=1 のみ）
    2. 目的変数（役職）のエンコード
    3. GroupShuffleSplit で訓練・テストデータ分割
    4. カテゴリカル特徴量のエンコード
    5. 話者確率特徴量の追加（combined_text が存在する場合）
    6. メタデータ（占い結果、追放者IDなど）の抽出
    7. 数値特徴量とカテゴリカル特徴量の分離・結合
    
    Parameters
    ----------
    csv_path : str
        読み込むCSVファイルのパス
    lang_feature : bool, default=False
        言語特徴量を使用するかどうか（将来の拡張用）
        
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
    
    # データの読み込みと基本フィルタリング
    df = pd.read_csv(csv_path)
    df = df[df["day"] == 1].copy()
    print(f"Successfully loaded '{csv_path}' and filtered for day 1.")
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
    groups = df['source_file'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(gss.split(df, y_full, groups=groups))
    except Exception as e:
        print(f"Error during GroupShuffleSplit. Do you have enough groups? Error: {e}")
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

    # エージェント予測確率特徴量の追加
    if 'combined_text' in df.columns:
        use_features = [
            "ReqDiscuss", "ReqListen", "Req(CO)", "Tally", "Admiration", "calm", "Wait", 
            "contradiction", "difficult", "confused", "Exe", "Atk", "Req(V)", "Req(T)", 
            "Pers", "Mt", "IF", "XOR"
        ]
        print("Adding speaker probability features...")
        df_train, df_test = get_speaker_probability_features(df_train, df_test, use_features)
        print("Speaker probability features added.")

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
        'combined_text', 'seer_co_order', 'alive'
    ]
    vote_cols = [c for c in df.columns if 'vote_id' in c]
    id_like_cols = [c for c in df.columns if c.endswith('_id')]
    flag_like_cols = [c for c in df.columns if c.endswith('_flag')]
    
    all_drop_cols = list(set(
        base_drop_cols + 
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
