"""
Data Creation Pipeline for Werewolf Role Prediction

このパイプラインは以下の処理を行います：
1. ログファイル（CSVまたはJSON）の読み込み
2. OpenAI APIを用いた発言タグ付け
3. 特徴量の抽出と集計
4. 最終的な特徴量テーブルの作成

処理フロー:
    Raw Logs → Tags (OpenAI) → Features (make_features.py) → Feature Table (CSV)
"""

import sys
import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# プロジェクトのsrcディレクトリをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# features モジュールのインポート
from features.tagger import process_csv, split_sentences_df
from features.make_features import create_table, aggregate_tags, summarize_tags, process_csv_files


def get_input_data_path(data_source: str = "spring") -> Path:
    """
    入力データディレクトリを取得
    
    Parameters
    ----------
    data_source : str, default="spring"
        データソース: "spring", "summer", "winter"
        
    Returns
    -------
    Path
        入力データディレクトリ
    """
    data_dir = PROJECT_ROOT / "data"
    source_map = {
        "spring": data_dir / "2025spring",
        "summer": data_dir / "2025summer",
        "winter": data_dir / "2024winter",
    }
    
    source_path = source_map.get(data_source)
    if not source_path or not source_path.exists():
        raise FileNotFoundError(f"Data source not found: {source_path}")
    
    return source_path


def get_output_paths(data_source: str = "spring") -> Tuple[Path, Path, Path]:
    """
    出力ディレクトリを取得または作成
    
    Parameters
    ----------
    data_source : str, default="spring"
        データソース名
        
    Returns
    -------
    tuple
        (tagged_output_dir, json_output_dir, feature_output_dir)
    """
    pipeline_dir = Path(__file__).parent
    project_root = pipeline_dir.parent.parent
    
    # 出力ディレクトリを作成
    tagged_dir = project_root / "data" / f"{data_source}_tagged"
    json_dir = project_root / "data" / f"{data_source}_json"
    feature_dir = project_root / "data" / f"{data_source}_features"
    
    for d in [tagged_dir, json_dir, feature_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    return tagged_dir, json_dir, feature_dir


def display_step(step_num: int, title: str) -> None:
    """
    ステップ情報を表示
    
    Parameters
    ----------
    step_num : int
        ステップ番号
    title : str
        ステップタイトル
    """
    print("\n" + "="*70)
    print(f"STEP {step_num}: {title}")
    print("="*70 + "\n")


def tag_csv_files(input_dir: Path, output_dir: Path) -> Tuple[int, int]:
    """
    CSVファイルにOpenAI APIを用いてタグを付与
    
    Parameters
    ----------
    input_dir : Path
        入力CSVディレクトリ
    output_dir : Path
        出力ディレクトリ
        
    Returns
    -------
    tuple
        (processed_count, error_count)
    """
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"⚠ Warning: No CSV files found in {input_dir}")
        return 0, 0
    
    processed = 0
    errors = 0
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        try:
            tagged_df, response_data = process_csv(str(csv_file))
            
            if not tagged_df.empty:
                # タグ付きCSVを保存
                output_file = output_dir / f"tagged_{csv_file.name}"
                tagged_df.to_csv(output_file, index=False, encoding="shift_jis")
                print(f"  ✓ Tagged data saved: {output_file.name}")
                processed += 1
            else:
                print(f"  ✗ Failed to tag: returned empty DataFrame")
                errors += 1
                
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}:")
            print(f"     {str(e)}")
            errors += 1
    
    print(f"\n✓ Tagging complete: {processed} processed, {errors} errors")
    return processed, errors


def create_features_from_tagged_csv(
    tagged_dir: Path, 
    json_output_dir: Path,
    feature_output_dir: Path
) -> Tuple[int, int]:
    """
    タグ付きCSVから特徴量を抽出
    
    Parameters
    ----------
    tagged_dir : Path
        タグ付きCSVディレクトリ
    json_output_dir : Path
        JSON出力ディレクトリ
    feature_output_dir : Path
        特徴量出力ディレクトリ
        
    Returns
    -------
    tuple
        (processed_count, error_count)
    """
    tagged_files = list(tagged_dir.glob("tagged_*.csv"))
    if not tagged_files:
        print(f"⚠ Warning: No tagged CSV files found in {tagged_dir}")
        return 0, 0
    
    processed = 0
    errors = 0
    
    print(f"Found {len(tagged_files)} tagged CSV files\n")
    
    for csv_file in tagged_files:
        filename = csv_file.name.replace("tagged_", "")
        print(f"Extracting features: {filename}")
        
        try:
            # タグの集計
            dict_list = aggregate_tags(str(csv_file))
            
            if not dict_list:
                print(f"  ✗ Failed to aggregate tags")
                errors += 1
                continue
            
            # 特徴量テーブルの作成
            feature_list = summarize_tags(dict_list)
            
            # JSONとして保存
            json_output_file = json_output_dir / filename.replace(".csv", ".json")
            result_dict = {
                "filename": filename,
                "agent_count": len(feature_list),
                "agents": feature_list
            }
            
            with open(json_output_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Features extracted: {json_output_file.name}")
            print(f"    - Agents: {len(feature_list)}")
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error extracting features from {filename}:")
            print(f"     {str(e)}")
            traceback.print_exc()
            errors += 1
    
    print(f"\n✓ Feature extraction complete: {processed} processed, {errors} errors")
    return processed, errors


def create_feature_table(
    json_dir: Path,
    output_csv_path: Path
) -> bool:
    """
    JSONファイルから最終的な特徴量テーブルを作成
    
    Parameters
    ----------
    json_dir : Path
        JSON特徴量ディレクトリ
    output_csv_path : Path
        出力CSV ファイルパス
        
    Returns
    -------
    bool
        成功したか
    """
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"⚠ Warning: No JSON files found in {json_dir}")
        return False
    
    print(f"Found {len(json_files)} JSON feature files\n")
    print("Creating comprehensive feature table...")
    
    try:
        all_rows = []
        
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            filename = data.get("filename", json_file.name)
            agents = data.get("agents", [])
            
            for agent_info in agents:
                row = {
                    "filename": filename,
                    "agent_id": agent_info.get("id"),
                    "role": agent_info.get("role"),
                    "agent_name": agent_info.get("agent_name"),
                    "character_name": agent_info.get("character_name"),
                    # Day 1 info
                    "day1_seer_co_order": agent_info.get("day1_info", {}).get("seer_co_order"),
                    "day1_seer_co_num": agent_info.get("day1_info", {}).get("seer_co_num"),
                }
                
                # 占い情報の追加
                if "day1_Div" in agent_info:
                    div1 = agent_info.get("day1_Div", {})
                    row["div1_agent"] = div1.get("agent")
                    row["div1_result"] = div1.get("result")
                
                if "day2_Div" in agent_info:
                    div2 = agent_info.get("day2_Div", {})
                    row["div2_agent"] = div2.get("agent")
                    row["div2_result"] = div2.get("result")
                
                # 投票情報
                row["day1_vote_id"] = agent_info.get("day1_vote_id")
                row["day2_vote_id"] = agent_info.get("day2_vote_id")
                
                # 発言特徴
                day1_info = agent_info.get("day1_info", {})
                day2_info = agent_info.get("day2_info", {})
                
                # Day1特徴
                for tag in ["Req(V)", "Req(T)", "Exe", "Agr", "Dis", "Sus", "Mt", "Pers"]:
                    row[f"day1_{tag}_count"] = len(day1_info.get(tag, []))
                
                for tag in ["CO", "Vote"]:
                    row[f"day1_{tag}_count"] = len(day1_info.get(tag, []))
                
                for tag in ["DivT", "Fact", "Est"]:
                    row[f"day1_{tag}_count"] = len(day1_info.get(tag, []))
                
                for tag in ["XorEst", "IF"]:
                    row[f"day1_{tag}_count"] = len(day1_info.get(tag, []))
                
                for tag in ["Req(CO)", "ReqDiscuss", "ReqListen", "Other", "calm", "Wait", 
                           "contradiction", "difficult", "confused", "admiration", "Tally", "Admiration"]:
                    row[f"day1_{tag}_count"] = len(day1_info.get(tag, []))
                
                all_rows.append(row)
        
        # DataFrameに変換
        df = pd.DataFrame(all_rows)
        
        # CSVに保存
        df.to_csv(output_csv_path, index=False, encoding="shift_jis")
        
        print(f"\n✓ Feature table created:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - Saved to: {output_csv_path.name}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating feature table:")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False


def display_pipeline_summary(
    data_source: str,
    tagged_count: int,
    feature_count: int,
    feature_table_created: bool
) -> None:
    """
    パイプライン実行結果のサマリーを表示
    
    Parameters
    ----------
    data_source : str
        データソース名
    tagged_count : int
        タグ付けされたファイル数
    feature_count : int
        特徴量抽出されたファイル数
    feature_table_created : bool
        特徴量テーブルが作成されたか
    """
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"\nData Source: {data_source.upper()}")
    print(f"Tagged Files: {tagged_count}")
    print(f"Feature Extracted: {feature_count}")
    print(f"Feature Table Created: {'✓ Yes' if feature_table_created else '✗ No'}")
    print("="*70 + "\n")


def main(data_source: str = "spring", skip_tagging: bool = False):
    """
    データ作成パイプラインのメイン処理
    
    Parameters
    ----------
    data_source : str, default="spring"
        データソース: "spring", "summer", "winter"
    skip_tagging : bool, default=False
        タグ付けステップをスキップするか（既にタグ付きCSVがある場合）
    """
    print("\n" + "="*70)
    print("WEREWOLF GAME - DATA CREATION PIPELINE")
    print("="*70 + "\n")
    
    try:
        # パス情報の取得
        input_dir = get_input_data_path(data_source)
        tagged_dir, json_dir, feature_dir = get_output_paths(data_source)
        
        print(f"Input Directory: {input_dir}")
        print(f"Output Directories:")
        print(f"  - Tagged: {tagged_dir}")
        print(f"  - JSON: {json_dir}")
        print(f"  - Features: {feature_dir}\n")
        
        tagged_count = 0
        feature_count = 0
        
        # ========== ステップ1: タグ付け ==========
        if not skip_tagging:
            display_step(1, "Tagging CSV Files with OpenAI API")
            print(f"Input directory: {input_dir}\n")
            tagged_count, tag_errors = tag_csv_files(input_dir, tagged_dir)
        else:
            print("\n⊘ Skipping tagging step (--skip-tagging flag)")
            tagged_count = len(list(tagged_dir.glob("tagged_*.csv")))
        
        # ========== ステップ2: 特徴量抽出 ==========
        display_step(2, "Extracting Features from Tagged CSV")
        feature_count, feature_errors = create_features_from_tagged_csv(
            tagged_dir, json_dir, feature_dir
        )
        
        # ========== ステップ3: 特徴量テーブル作成 ==========
        display_step(3, "Creating Comprehensive Feature Table")
        
        # 出力CSVファイル名を決定
        csv_filename = f"all_feature_table_{data_source}.csv"
        output_csv = feature_dir / csv_filename
        
        feature_table_created = create_feature_table(json_dir, output_csv)
        
        # ========== 完了表示 ==========
        display_pipeline_summary(data_source, tagged_count, feature_count, feature_table_created)
        
        if feature_table_created:
            print(f"✓ PIPELINE COMPLETED SUCCESSFULLY")
            print(f"Final CSV: {output_csv}\n")
        else:
            print(f"⚠ Pipeline completed with errors. Check logs above.\n")
        
    except Exception as e:
        print(f"\n✗ PIPELINE FAILED:")
        print(f"  {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data creation pipeline for Werewolf role prediction"
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
