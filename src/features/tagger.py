import os
import pandas as pd
import openai
import json
from tqdm import tqdm
import glob
from openai import OpenAI
import pandas as pd
import itertools
import re

def split_sentences(text):
    # 文末の句点（。！？）で分割
    if pd.isnull(text):
        return []
    sentences = re.split(r'(?<=[。！？])', text)
    # 空文字列を除去しstrip
    return [s.strip() for s in sentences if s.strip()]

def split_sentences_df(df):
    other_info =df[df["type"] != "talk"]
    talk = df[df["type"] == "talk"]
    talk = talk[talk["info4"] != "Over"]
    talk = talk.dropna(subset=["info4"])
    talk_expanded = talk.apply(
        lambda row: [
            {**row.to_dict(), "info4": sentence}
            for sentence in split_sentences(row["info4"])
        ],
        axis=1
    )
    talk_expanded = pd.DataFrame(list(itertools.chain.from_iterable(talk_expanded)))
    talk_expanded=pd.concat([talk_expanded,other_info], ignore_index=True, sort=False)
    
    return talk_expanded


def clean_json_trailing_commas(json_str):
    # 末尾のカンマを削除（配列の直前）
    json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)
    # 「<br>」という文字列も削除
    json_str = json_str.replace("<br>", "")
    return json_str

# テスト用
# OpenAI API呼び出し（新API対応）
def tag_with_openai(utterance, tagging_guideline=""):
    API_key=os.getenv("OPENAI_API_KEY")#環境変数に登録してあるAPIキーを取得
    client = OpenAI(api_key=API_key) 

    if tagging_guideline == "":
        with open("prompt.txt", "r", encoding="utf-8") as f:
            tagging_guideline = f.read()

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": tagging_guideline},
                {"role": "user", "content": f"Based on the rules above, please tag the following data. \ncontents: {json.dumps(utterance, ensure_ascii=False)}"}
            ],
            response_format={"type": "json_object"},
            temperature=1
        )
        
        # content の取り出し
        content = response.choices[0].message.content
        #print(content)
        return content
    
    except Exception as e:
        print(f"[ERROR] Failed to process utterance: {utterance}")
        print(f"[ERROR] Exception: {e}")
        return {"tags": []}


# CSVを読み込んでタグを付与
def process_csv(input_path):
    flag_1day= False
    try:
        #df = pd.read_csv(input_path, encoding="shift_jis",names=["date","type", "info1", "info2", "info3", "info4"])
        df = pd.read_csv(input_path,names=["date","type", "info1", "info2", "info3", "info4","info5"])
    except :
        #df = pd.read_csv(input_path, encoding="shift_jis",names=["date","type", "info1", "info2", "info3", "info4","tag"])
        df = pd.read_csv(input_path,names=["date","type", "info1", "info2", "info3", "info4","info5","tag"])
    filename= os.path.basename(input_path)
    #print(f"Processing file: {filename}")

    status=df[(df["type"]=="status")&(df["date"]==0)][["info1","info5"]].values
    status_dict={k:v for k,v in status}

    target_df= split_sentences_df(df)# 1文ずつに分割
    try:
        target_df["info4"] = target_df.apply( # typeが"talk"ならinfo3の数字を文字列の先頭に追加
            lambda row: f"{status_dict[int(row['info3'])]}:{row['info4']}" if row["type"] == "talk" and pd.notnull(row["info3"]) and pd.notnull(row["info4"]) else row["info4"],
            axis=1)
    except Exception as e:
        print(f"[ERROR] Failed to process DataFrame: {input_path}")
        return pd.DataFrame([]), []
    taged_list = []
    taged_df_list = []
    
    for d in range(3):
        target1= target_df.copy()
        target1 = target1[(target1["date"] == d) & (target1["type"] == "talk")]["info4"]
        target1_df=pd.DataFrame(target1)
        if (target1.empty) and (d==2):
            #print(f"{d}日目の発言がありません。")
            flag_1day = True
            continue
        elif target1.empty:
            #print(f"{d}日目の発言がありません。不明なエラーです。")
            continue

        #発言を20個ずつに分割
        target_list=[]
        divide_num=len(target1) // 20 # 20個ずつに分割するための回数.小数点以下を切り捨て
        if len(target1) % 20 != 0:
            divide_num = int(divide_num) + 1  # 20に分割し、余りがある場合は1回多く回す
        #print(f"{d}日目の発言数: {len(target1)}")
        #print(f"{d}日目の発言を20個ずつに分割します。{divide_num}回")

        start_index = 0
        for i in range(divide_num):
            end_index = start_index + 20
            if end_index > len(target1):#20で分割した余りの処理。
                end_index = len(target1)
            target_part = target1[start_index:end_index].tolist()
            target_list.append(target_part)
            start_index += 20

        target_tagged_list=[]
        sub_taged_tagged_list = []

        print(target_list)
        # タグ付け関数を呼び出し
        for i, target in enumerate(target_list):
            taged_data = tag_with_openai(target)
            #print(type(taged_data))
            taged_data = json.loads(taged_data)
            #print("taged_data:", taged_data)
            taged_data = taged_data["tags"]
            target_tagged_list.append(taged_data)
            if len(taged_data) > len(target):
                taged_data = taged_data[:len(target)]
                sub_taged_tagged_list.append(taged_data)
                print(f"Warning: {d}日目の{i+1}回目のタグ付けで、タグの数が発言数より多かったため、後ろよりのタグを削除しました。")
            elif len(taged_data) < len(target):
                # targetと同じ要素数まで空白を入れる
                sub_taged_tagged_list.append(taged_data + [[] for _ in range(len(target) - len(taged_data))])
                print(f"Warning: {d}日目の{i+1}回目のタグ付けで、タグの数が発言数より少なかったため、空白を追加しました。")
            else:
                sub_taged_tagged_list.append(taged_data)
            print("targed_data:", taged_data)

        # 結果を1つのリストにまとめる
        for part in target_tagged_list:
            taged_list.append(part)

        
        #sub_taged:入力と出力の長さが違うと、dfに連結できないので同じ長さのリストを作成しておく
        #dfと連結できないエラーを発生させたくないので、空白で埋める。ただ、APIの返答を記録したいのでtaged_dataも作っておく。後でtxtファイルに保存する。
        sub_taged_data = []
        for part in sub_taged_tagged_list:
            sub_taged_data.extend(part)

        try:
            target1_df["tag"]= sub_taged_data  # タグをDataFrameに追加
            taged_df_list.append(target1_df)

        except Exception as e:
            print(f"Error adding tags to DataFrame: {e}")
            return pd.DataFrame([]), taged_list
            
    try:
        filename= os.path.basename(input_path)
        # タグ付け結果をCSVに保存
        if flag_1day:
            #print("２日目の発言がありません。１日目のみのデータを保存します。")
            concat_df = taged_df_list[0]
        else:
            concat_df=pd.concat([taged_df_list[0], taged_df_list[1]], axis=0, ignore_index=True)
        
        concat_df_copy= concat_df.copy()
        concat_full_df = pd.merge(target_df, concat_df_copy, on="info4", how="left")
        
        #dfを保存
        output_path1 = "taged_data2"
        if not os.path.exists(output_path1):
            os.makedirs(output_path1)
        output_file = os.path.join(output_path1, f"taged_{filename}.csv")
        concat_df.to_csv(output_file, index=False, encoding="shift_jis")

        output_path2="taged_data(特徴量作成用)2"
        if not os.path.exists(output_path2):
            os.makedirs(output_path2)
        output_file = os.path.join(output_path2, f"taged_{filename}.csv")
        concat_full_df.to_csv(output_file, index=False, encoding="shift_jis")

        # APIのレスポンスをtxtファイルに保存
        output_path3= "API_responce2"
        if not os.path.exists(output_path3):
            os.makedirs(output_path3)
        log_output_file = os.path.join(output_path3, f"taged_{filename}.txt")
        with open(log_output_file, "w", encoding="shift_jis") as f:
            f.write(str(taged_list))
  
        return concat_df,target_df
    except Exception as e:
        #print(f"Error concatenating DataFrames: {e}")
        return pd.DataFrame([]),taged_list