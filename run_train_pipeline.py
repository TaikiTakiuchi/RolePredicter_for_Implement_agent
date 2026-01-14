"""
Execution script for the training pipeline

このスクリプトはプロジェクトのルートディレクトリから実行してください：
    python run_pipeline.py
"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from src.pipelines.training_pipeline import main as training_main

if __name__ == "__main__":
    predictor = training_main()
