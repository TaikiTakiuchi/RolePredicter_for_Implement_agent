# RolePredicter - 人狼ゲーム役職推定システム

このプロジェクトは、XGBoost と Optuna を用いた人狼ゲームの役職推定システムです。複数の視点（村人、占い師、人狼）から役職を予測します。

## 📋 目次

- [概要](#概要)
- [必要な環境](#必要な環境)
- [プロジェクト構成](#プロジェクト構成)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [詳細な使用方法](#詳細な使用方法)
- [モデル仕様](#モデル仕様)
- [トラブルシューティング](#トラブルシューティング)

---

## 概要

このシステムは、人狼ゲームの実際のプレイログから特徴量を抽出し、XGBoost を用いて複数の視点から役職を推定します。

**主な特徴：**
- 🎯 3つの異なる視点から役職を予測（村人視点、占い師視点、人狼視点）
- 🔧 ハイパーパラメータ最適化（Optuna による自動チューニング）
- 📊 ゲームルール制約を組み込んだ役職割り当て
- 📈 複数モデル管理による統合予測

---

## 必要な環境

**Python バージョン:**
- Python 3.8 以上

**主要な依存パッケージ:**
```
xgboost>=1.5.0
optuna>=2.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
joblib>=1.0.0
```

---

## プロジェクト構成

```
RolePredicter(GitHub投稿準備中)/
├── README.md                          # このファイル
├── run_train_pipeline.py              # 訓練パイプライン実行スクリプト
├── run_data_pipeline.py               # データ作成パイプライン実行スクリプト
├── src/
│   ├── Rolepredicter/
│   │   ├── __init__.py
│   │   ├── data_preparation.py        # データ準備・前処理
│   │   ├── role_assignment.py         # 役職割り当てロジック
│   │   └── role_predictor.py          # メインの予測クラス
│   ├── features/
│   │   ├── make_features.py           # 特徴量抽出・集計
│   │   ├── tagger.py                  # OpenAI APIによるタグ付け
│   │   └── prompt.txt                 # タグ付けプロンプト
│   └── pipelines/
│       ├── __init__.py
│       ├── training_pipeline.py       # 訓練パイプライン
│       └── data_creation_pipeline.py  # データ作成パイプライン
├── data/                              # データディレクトリ
│   ├── 2024winter/                    # 冬季ログ
│   ├── 2025spring/                    # 春季ログ
│   └── 2025summer/                    # 夏季ログ
├── models/                            # 学習済みモデル保存先
└── notebooks/                         # Jupyter ノートブック
```

---

## インストール

### 1. リポジトリをクローン（または解凍）
```bash
cd RolePredicter(GitHub投稿準備中)
```

### 2. 必要なパッケージをインストール
```bash
pip install -r requirements.txt
```

または個別にインストール：
```bash
pip install xgboost optuna scikit-learn pandas numpy joblib
```

### 3. データの配置
訓練用の CSV ファイルを `data/` ディレクトリに配置してください。

```
data/
├── 2024winter/
├── 2025spring/
└── 2025summer/
    └── all_feature_table_2025sm.csv  # ← このような形式
```

---

## クイックスタート

このプロジェクトには 2 つのメインパイプラインがあります。

### パイプライン1: データ作成（初回のみ必須）

ログファイルから特徴量テーブルを作成します。

```bash
# 春季データを処理
python run_data_pipeline.py --source spring

# 夏季データを処理
python run_data_pipeline.py --source summer

# タグ付けステップをスキップ（既にタグ付きCSVがある場合）
python run_data_pipeline.py --source spring --skip-tagging
```

**処理内容:**
1. ✅ OpenAI APIを用いた発言のタグ付け
2. ✅ タグの集計と特徴量抽出
3. ✅ 最終的な特徴量テーブル作成（CSV）

**出力:**
```
data/
├── spring_tagged/          # タグ付きCSV
├── spring_json/            # JSON形式の特徴量
└── all_feature_table_spring.csv  # 最終出力
```

### パイプライン2: モデル訓練

作成された特徴量テーブルからモデルを訓練します。

```bash
python run_train_pipeline.py
```

**処理内容:**
1. ✅ 特徴量テーブルの読み込みと前処理
2. ✅ 3つのモデル（村人、占い師、人狼視点）の訓練
3. ✅ Optuna によるハイパーパラメータ最適化
4. ✅ 予測結果の評価と表示
5. ✅ モデルの自動保存（`models/` に保存）

**出力:**
```
models/
├── human_model.joblib
├── seer_model.joblib
└── werewolf_model.joblib
```

---

## 詳細な使用方法

### データ作成パイプラインの詳細

#### 基本的な実行
```bash
# デフォルト（春季データ）
python run_data_pipeline.py

# 特定の季節を指定
python run_data_pipeline.py --source winter
python run_data_pipeline.py --source summer
```

#### 必須環境：OpenAI API キー
```bash
# PowerShell
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxx"

# コマンドプロンプト
set OPENAI_API_KEY=sk-xxxxxxxxxxxx

# Linux/Mac
export OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

#### パイプラインの流れ
```
Raw Logs (CSV)
    ↓
[STEP 1] OpenAI APIでタグ付け（features/tagger.py）
    ↓
Tag#SV
    ↓
[STEP 2] 特徴量抽出（features/make_features.py）
    ↓
JSON特徴量ファイル
    ↓
[STEP 3] 最終的な特徴量テーブル作成
    ↓
all_feature_table_[season].csv
```

### モデル訓練パイプラインの詳細

#### 基本的な実行
```bash
python run_train_pipeline.py
```

#### Python スクリプトから直接使用

```python
from src.Rolepredicter.role_predictor import RolePredictor

# 1. 初期化（データの準備）
predictor = RolePredictor('data/all_feature_table_2025sm.csv', lang_feature=True)

# 2. モデルの訓練
predictor.train(n_trials=200)  # 200回のハイパーパラメータ探索

# 3. 予測（確率）
probs = predictor.predict('human', X_new)
# 出力: (n_samples, 4) の確率行列

# 4. 予測（ラベル）
labels = predictor.predict_label('seer', X_new)
# 出力: 予測された役職のID

# 5. 予測（役職名）
role_names = predictor.predict_role_names('werewolf', X_new)
# 出力: ['WEREWOLF', 'VILLAGER', ...]

# 6. モデルの保存
predictor.save_model('human', 'models/human_model.joblib')

# 7. モデルの読み込み
predictor.load_model('human', 'models/human_model.joblib')
```

### 方法2: Jupyter Notebook から使用

```python
# notebooks/ フォルダに .ipynb ファイルを作成して実行
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

from Rolepredicter.role_predictor import RolePredictor

# 以下、方法1と同じ
predictor = RolePredictor('data/all_feature_table_2025sm.csv')
predictor.train(n_trials=100)
```

#### データのみ準備する場合

```python
from src.Rolepredicter.data_preparation import prepare_data_for_training_with_meta

# データの準備と分割
X_train, X_test, y_train, y_test, meta_train, meta_test, \
label_encoder, feature_names, _, _ = prepare_data_for_training_with_meta(
    'data/all_feature_table_2025sm.csv',
    lang_feature=True
)

print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
```

---

## モデル仕様

### 3つの異なる視点モデル

#### 1. **村人視点モデル** (`human`)
- 自分が村人（通常や取憑き）の立場から役職を推定
- **評価指標**: 人狼の検出精度 (F1-score)
- **用途**: 村人プレイヤーの意思決定支援

#### 2. **占い師視点モデル** (`seer`)
- 自分が占い師の立場から、占い結果を組み込んで役職を推定
- **特徴**: 黒判定（人狼）と白判定（人間）で結果を分類
- **評価指標**: F1-score（黒・白別に評価）
- **用途**: 占い師プレイヤーの推理支援

#### 3. **人狼視点モデル** (`werewolf`)
- 自分が人狼の立場から他プレイヤーの役職を推定
- **評価指標**: 取憑きの検出精度 (F1-score)
- **用途**: 人狼チームの戦略立案支援

### ゲーム設定

- **プレイヤー数**: 5 名
- **役職分布**: POSSESSED (1), SEER (1), VILLAGER (2), WEREWOLF (1)
- **対象**: Day 1 のデータのみ

### ハイパーパラメータ最適化

Optuna を用いて以下を自動チューニング：
- `max_depth`: 木の最大深さ (3-10)
- `lambda`: L2 正則化 (1e-8 ~ 10.0)
- `alpha`: L1 正則化 (1e-8 ~ 10.0)
- `learning_rate`: 学習率 (0.01 ~ 0.3)
- `n_estimators`: 決定木の数 (100 ~ 2000)
- その他: `gamma`, `min_child_weight`, `subsample`, `colsample_bytree`

---

## 主要な関数・メソッド

### RolePredictor クラス

| メソッド | 説明 | 戻り値 |
|---------|------|--------|
| `__init__(csv_path, lang_feature)` | 初期化＆データ準備 | なし |
| `train(n_trials)` | 3つのモデルを訓練 | なし |
| `predict(model_name, X)` | 確率予測 | (n_samples, 4) array |
| `predict_label(model_name, X)` | ラベル予測 | (n_samples,) array |
| `predict_role_names(model_name, X)` | 役職名予測 | List[str] |
| `save_model(model_name, path)` | モデル保存 | なし |
| `load_model(model_name, path)` | モデル読み込み | なし |

### データ準備関数

```python
from src.Rolepredicter.data_preparation import prepare_data_for_training_with_meta

result = prepare_data_for_training_with_meta(csv_path, lang_feature=False)
```

**戻り値:**
- `X_train`: 訓練特徴量 (n_train, n_features)
- `X_test`: テスト特徴量 (n_test, n_features)
- `y_train`: 訓練ラベル (n_train,)
- `y_test`: テストラベル (n_test,)
- `meta_train`: 訓練メタデータ (占い結果など)
- `meta_test`: テストメタデータ
- `label_encoder`: 役職エンコーダー
- `feature_names`: 特徴量名リスト

---

## 結果の解釈

### 出力例

```
======================================================================
WEREWOLF ROLE PREDICTION - TRAINING PIPELINE
======================================================================

STEP 1: Data Preparation
----------------------------------------------------------------------
✓ Data file selected: all_feature_table_2025sm.csv
✓ Data prepared:
  - Training: (1000, 128)
  - Test: (250, 128)
✓ Roles: ['POSSESSED', 'SEER', 'VILLAGER', 'WEREWOLF']
✓ Features: 128

STEP 2: Model Training
======================================================================

--- Training HUMAN Perspective Model ---
[############################################] 200/200 trials
Best parameters: {'max_depth': 6, 'lambda': 0.5, ...}
Best F1-score: 0.7823

✓ Model saved to models/human_model.joblib

[訓練完了...]
```

### 精度の目安

- **F1-score 0.70 以上**: 良好（実用的）
- **F1-score 0.50～0.70**: 改善余地あり
- **F1-score 0.50 未満**: 要改善（特徴量の見直し推奨）

---

## トラブルシューティング

### Q: "No CSV files found in data directory" エラー
**A:** `data/` ディレクトリに CSV ファイルが配置されているか確認してください。

```bash
ls data/
```

### Q: メモリ不足エラー
**A:** `n_trials` を減らすか、Optuna の並列実行を調整してください。

```python
predictor.train(n_trials=50)  # 200 → 50 に削減
```

### Q: 予測精度が低い
**A:** 以下を試してください：
1. データの品質確認（外れ値、欠損値の確認）
2. 特徴量の追加（`lang_feature=True` で言語特徴を有効化）
3. ハイパーパラメータの手動調整

### Q: モデル読み込み時のエラー
**A:** モデルが保存されているか確認：

```bash
ls models/
# human_model.joblib, seer_model.joblib, werewolf_model.joblib が表示されるはず
```

---

## 今後の拡張予定

- [ ] Day 2 以降のデータ対応
- [ ] より高度なゲーム制約の組み込み
- [ ] オンライン学習機能
- [ ] Web API 化
- [ ] 予測確信度の可視化

---

## ライセンス

このプロジェクトは瀧内の研究用です。

---

## 作成者・更新日

作成日: 2026 年 1 月  
最終更新: 2026 年 1 月 14 日
