"""
Execution script for the data creation pipeline

このスクリプトはプロジェクトのルートディレクトリから実行してください：
    python run_data_pipeline.py [--source spring|summer|winter] [--skip-tagging]

例：
    python run_data_pipeline.py --source spring
    python run_data_pipeline.py --source summer --skip-tagging
"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from src.pipelines.data_creation_pipeline import main

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run data creation pipeline"
    )
    parser.add_argument(
        "--source",
        choices=["spring", "summer", "winter"],
        default="spring",
        help="Data source season (default: spring)"
    )
    parser.add_argument(
        "--skip-tagging",
        action="store_true",
        help="Skip tagging step if tagged CSVs already exist"
    )
    
    args = parser.parse_args()
    main(data_source=args.source, skip_tagging=args.skip_tagging)
