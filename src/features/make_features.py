import pandas as pd
import glob
import ast
import matplotlib.pyplot as plt
import japanize_matplotlib
import re
import os


# 安全にtag列を評価し、カンマ分割補正（ただしXorEst含む場合は除外）
def safe_eval_and_split(x):
    try:
        if not isinstance(x, str):
            return []
        val = ast.literal_eval(x)
        # 分割条件: 要素1、カンマを含む、XorEstを含まない
        if (
            isinstance(val, list)
            and len(val) == 1
            and isinstance(val[0], str)
            and "," in val[0]
            and "XorEst" not in val[0]
        ):
            #print(val)
            return [t.strip() for t in val[0].split(',')]
        elif isinstance(val, list):
            return val
        else:
            return []
    except Exception:
        return []
    

def create_table(path):
    # CSV読み込み
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8")
    except Exception as e:
        df = pd.read_csv(
            path,
            encoding="shift-jis")
    # タグをリスト化
    tag_split = df["tag"].apply(safe_eval_and_split)
    # 最大タグ数を計算
    max_tags = (tag_split.dropna()
        .apply(lambda x: len(x) if isinstance(x, list) else 0)
        .max())
    # tag_1, tag_2, ..., tag_n を列に展開
    tag_df = pd.DataFrame(tag_split.tolist(), columns=[f"tag_{i+1}" for i in range(max_tags)])
    # 元データと結合
    concat_df = pd.concat([df, tag_df], axis=1)
    # 特定タグを正規化（Req(Discuss) → ReqDiscuss, Req(listen) → ReqListen）
    for i in range(1, max_tags + 1):
        col = f"tag_{i}"
        concat_df[col] = concat_df[col].apply(
            lambda x: "ReqDiscuss" if isinstance(x, str) and x.startswith("Req(Discuss)") else x
        )
        concat_df[col] = concat_df[col].apply(
            lambda x: "ReqListen" if isinstance(x, str) and x.startswith("Req(listen)") else x
        )
    idname_dict = df[(df["type"] == "status")&(df["date"]==0)].set_index("info1")["info5"].to_dict()
    #print("idname_dict:", idname_dict)
    return concat_df, idname_dict,max_tags

def aggregate_tags(path):
    print(f"Processing file: {path}")
    df, idname_dict,max_tags = create_table(path)

    #読み込めないとめんどいので型変換
    df["date"] = df["date"].astype(int)
    df["info1"] = df["info1"].astype(int)

    if max_tags == 0:
        print("タグが全て空、もしくは読み取れませんでした。path:", path)
        return []
    
    df["info3"] = df["info3"].astype(str)
    dict_list = []
    target_talk = df[df["date"].isin([0, 1, 2]) & (df["type"] == "talk")]
    # --- CO順の検出処理 ---
    co_order_dict = {}
    co_count = 1
    for _, row in target_talk.iterrows():
        speaker_id = int(row["info3"])  # 発言者のID（1〜5）
        speaker_name = idname_dict.get(speaker_id, "")
        for j in range(1, max_tags + 1):
            tag_col = f"tag_{j}"
            tag_value = row.get(tag_col)
            if isinstance(tag_value, str):
                #print("check")
                match = re.fullmatch(r"CO\[(.*)\]:占い師", str(tag_value))
                if match:
                    co_tag_name = str(match.group(1))  # タグ内の名前
                    #print(f"CO detected: {co_tag_name} (Speaker ID: {speaker_id})")
                    #print(f"Speaker Name: {speaker_name}")
                    if co_tag_name == speaker_name:
                        if speaker_id not in co_order_dict:
                            co_order_dict[speaker_id] = co_count
                            co_count += 1

    # --- エージェントごとの情報を収集 ---
    #占い師情報取得
    target_talk0 = df[(df["date"] == 0)]
    target_talk1 = df[(df["date"] == 1)]
    target_talk2 = df[(df["date"] == 2)]
    try:
        df["date"] = df["date"].astype(int)
        Div1_agent = df[(df["type"] == "divine") & (df["date"] == 0)]["info2"].values[0]
        Div1_result = df[(df["type"] == "divine") & (df["date"] == 0)]["info3"].values[0]
        print("Div1_agent:", Div1_agent, "Div1_result:", Div1_result)
    except IndexError:
        print("1日目の占い情報が見つかりませんでした")
        Div1_agent, Div1_result = None, None
    try:
        Div2_agent = df[(df["type"] == "divine") & (df["date"] == 1)]["info2"].values[0]
        Div2_result = df[(df["type"] == "divine") & (df["date"] == 1)]["info3"].values[0]
    except IndexError:
        print("2日目の占い情報が見つかりませんでした")
        Div2_agent, Div2_result = None, None

    # エージェントごとにタグを集計
    for i in range(1, 6):  # Agent 1〜5
        agent_rows0  = target_talk0[target_talk0["info3"] == str(i)]
        agent_rows1 = target_talk1[target_talk1["info3"] == str(i)]
        agent_rows2 = target_talk2[target_talk2["info3"] == str(i)]
        #投票情報取得
        try:
            vote1_agent = df[(df["type"] == "vote") & (df["date"] == 1)]["info2"].values[i-1]
        except IndexError:
            print("1日目の投票情報が見つかりませんでした")
            vote1_agent = None
        try:
            vote2_agent = df[(df["type"] == "vote") & (df["date"] == 2)]["info2"].values[i-1]
        except IndexError:
            print("投票情報が見つかりませんでした")
        vote2_agent = None
        talk_list0, talk_list1, talk_list2 = [], [], []
        for j in range(1, max_tags + 1):
            col = f'tag_{j}'
            if col in agent_rows0.columns:
                talk_list0.extend(tag for tag in agent_rows0[col].dropna() if isinstance(tag, str))
            if col in agent_rows1.columns:
                talk_list1.extend(tag for tag in agent_rows1[col].dropna() if isinstance(tag, str))
            if col in agent_rows2.columns:
                talk_list2.extend(tag for tag in agent_rows2[col].dropna() if isinstance(tag, str))

        status_rows = df[(df["type"] == "status") & (df["date"] == 0) & (df["info1"] == i)]
        #print(df[(df["type"] == "status") & (df["date"] == 0)])
        if status_rows.empty:
            #print(df.dtypes)
            print(f"Agent {i} のstatus情報が見つかりませんでした")
            continue
        role = status_rows["info2"].values[0]
        agent_name = status_rows["info4"].values[0]
        character_name = status_rows["info5"].values[0]
        co_order = co_order_dict.get(i, None)
        info_dict={
            "id": i,
            "role": role,
            "agent_name": agent_name,
            "character_name": character_name,
            "day0_talks": talk_list0,
            "day1_talks": talk_list1,
            "day1_vote_id": vote1_agent,
            "day2_vote_id": vote2_agent,
            "day2_talks": talk_list2,
            "seer_co_order": co_order,
            "seer_co_num": co_count-1
        }
        if role=="SEER":
            info_dict["day1_Div"]={"agent": Div1_agent, "result": Div1_result}
            info_dict["day2_Div"]={"agent": Div2_agent, "result": Div2_result}
        else:
            info_dict["day1_Div"]={"agent": None, "result": None}
            info_dict["day2_Div"]={"agent": None, "result": None}

        dict_list.append(info_dict)
    return dict_list

def summarize_tags(dict_list):
    category1 = ["Req(V)", "Req(T)", "Exe", "Agr", "Dis", "Sus", "Mt", "Pers"]
    category2 = ["CO", "Vote"]
    category3 = ["DivT", "Fact", "Est"]
    category4 = ["XorEst", "IF"]
    category5 = ["Req(CO)", "ReqDiscuss", "ReqListen", "Other", "calm", "Wait", "contradiction", "difficult", "confused", "admiration", "Tally","Admiration"]
    summarize_list = []

    for Dict in dict_list:
        agent_id, role, agent_name, character_name,Div1,Div2,vote1,vote2 = Dict["id"], Dict["role"], Dict["agent_name"], Dict["character_name"],Dict["day1_Div"],Dict["day2_Div"],Dict["day1_vote_id"],Dict["day2_vote_id"]
        talk_list= [Dict["day0_talks"]+ Dict["day1_talks"], Dict["day2_talks"]] #day0とday1は連結させる, day2は別
        info_dict = {"id": agent_id, "role": role, 
                     "agent_name": agent_name, 
                     "character_name": character_name, 
                     "day1_info": {"seer_co_order":Dict["seer_co_order"],"seer_co_num":Dict["seer_co_num"]},
                     "day2_info": {},
                     "day1_Div": Div1,
                     "day2_Div": Div2,
                     "day1_vote_id": vote1,
                     "day2_vote_id": vote2
                     }
        for i, talks in enumerate(talk_list, start=1):
            day_info = info_dict[f"day{i}_info"]
            for t in talks:
                if ":" in t:
                    # category1
                    if any(t.startswith(cat) for cat in category1):
                        try:
                            tag, recepient_id = t.split(":", 1)
                            if tag in ["Agr","Dis"]:
                                if "(" in recepient_id:
                                    recepient_id = recepient_id.split("(")[0].strip()
                                    print("recepient_id edit :", recepient_id)
                            if tag not in day_info:
                                day_info[tag] = [{"recepient_id": recepient_id, "count": 1}]
                                continue
                            found = False
                            for entry in day_info[tag]:
                                if entry["recepient_id"] == recepient_id:
                                    entry["count"] += 1
                                    found = True
                                    break
                            if not found:
                                day_info[tag].append({"recepient_id": recepient_id, "count": 1})
                        except Exception as e:
                            print("error1:想定外のタグです:", t, e)
                        continue
                    # category2
                    if any(t.startswith(cat) for cat in category2):
                        print("category2:", t)
                        try:
                            tag_and_speakerid, recepient_role = t.split(":")
                            tag, speaker_id = tag_and_speakerid.split("[")
                            speaker_id = speaker_id.rstrip("]")
                        except Exception as e:
                            print("error2: タグのパースに失敗しました:", t, e)
                            continue
                        if tag not in day_info:
                            day_info[tag] = [{"speaker_id": speaker_id, "recepient_role": recepient_role, "count": 1}]
                            continue
                        found = False
                        for entry in day_info[tag]:
                            if entry["recepient_role"] == recepient_role and entry["speaker_id"] == speaker_id:
                                entry["count"] += 1
                                found = True
                                break
                        if not found:
                            day_info[tag].append({"speaker_id": speaker_id, "recepient_role": recepient_role, "count": 1})
                        continue
                    # category3
                    if any(t.startswith(cat) for cat in category3):
                        try:
                            tag_and_speakerid, recepient_info = t.split(":")
                            tag, speaker_id = tag_and_speakerid.split("[")
                            speaker_id = speaker_id.rstrip("]")
                            recepient_id, recepient_role = recepient_info.split("->")
                            tag, speaker_id, recepient_id, recepient_role = tag.strip(), speaker_id.strip(), recepient_id.strip(), recepient_role.strip()
                        except Exception as e:
                            print("error3: タグのパースに失敗しました:", t, e)
                            continue
                        if tag not in day_info:
                            day_info[tag] = [{"speaker_id": speaker_id, "recepient_id": recepient_id, "recepient_role": recepient_role, "count": 1}]
                            continue
                        found = False
                        for entry in day_info[tag]:
                            if entry["recepient_role"] == recepient_role and entry["speaker_id"] == speaker_id :
                                entry["count"] += 1
                                found = True
                                break
                        if not found:
                            day_info[tag].append({"speaker_id": speaker_id, "recepient_id": recepient_id, "recepient_role": recepient_role, "count": 1})
                        continue
                    # category4
                    if t.startswith("XorEst"):
                        try:
                            tag_and_ids, roles = t.split("->")
                            tag, ids = tag_and_ids.split("(")
                            tag = tag.strip().rstrip(":")
                            ids = ids.rstrip(")")
                            id1, id2 = [x.strip() for x in ids.split(",")]
                            roles = roles.strip().lstrip("(").rstrip(")")
                            role1, role2 = [x.strip() for x in roles.split("⇔")]
                        except Exception as e:
                            print("error4: タグのパースに失敗しました:", t, e)
                            continue
                        if tag not in day_info:
                            day_info[tag] = [{"id1": id1, "id2": id2, "role1": role1, "role2": role2, "count": 1}]
                            continue
                        found = False
                        for entry in day_info[tag]:
                            if entry["id1"] == id1 and entry["id2"] == id2 and entry["role1"] == role1 and entry["role2"] == role2:
                                entry["count"] += 1
                                found = True
                                break
                        if not found:
                            day_info[tag].append({"id1": id1, "id2": id2, "role1": role1, "role2": role2, "count": 1})
                        continue
                    if t.startswith("IF"):
                        if not t.startswith("IF:"):
                            t = t.replace("IF", "IF:", 1)
                        try:
                            tag_and_left, right = t.split("->")
                            tag, left = tag_and_left.split(":", 1)
                            left = left.strip().lstrip("(").rstrip(")")
                            right = right.strip()
                            left_id, left_role = [s.strip() for s in left.split(":", 1)]
                            right_id, right_role = [s.strip() for s in right.split(":", 1)]
                        except Exception as e:
                            print(f"解析エラー: {e} / 入力: {t}")
                            continue
                        if tag not in day_info:
                            day_info[tag] = [{
                                "left_id": left_id, "left_role": left_role,
                                "right_id": right_id, "right_role": right_role,
                                "count": 1
                            }]
                            continue
                        found = False
                        for entry in day_info[tag]:
                            if entry["left_id"] == left_id and entry["left_role"] == left_role and entry["right_id"] == right_id and entry["right_role"] == right_role:
                                entry["count"] += 1
                                found = True
                                break
                        if not found:
                            day_info[tag].append({
                                "left_id": left_id, "left_role": left_role,
                                "right_id": right_id, "right_role": right_role,
                                "count": 1
                            })
                        elif t.startswith("IF"):
                            # IFの次の文字が:でなければ:を追加
                            if not t.startswith("IF:"):
                                t = t.replace("IF", "IF:", 1)
                            try:
                                tag_and_left, right = t.split("->")
                                tag, left = tag_and_left.split(":", 1)
                                left = left.strip().lstrip("(").rstrip(")")
                                right = right.strip().lstrip("(").rstrip(")")
                                left_id, left_role = [s.strip() for s in left.split(":", 1)]
                                right_id, right_role = [s.strip() for s in right.split(":", 1)]
                            except Exception as e:
                                print(f"解析エラー: {e} / 入力: {t}")
                                continue
                            if tag in info_dict[f"day{i}_info"]:
                                found = False
                                if isinstance(info_dict[f"day{i}_info"][tag], list):
                                    for entry in info_dict[f"day{i}_info"][tag]:
                                        if entry["left_id"] == left_id and entry["left_role"] == left_role and \
                                        entry["right_id"] == right_id and entry["right_role"] == right_role:
                                            entry["count"] += 1
                                            found = True
                                            break
                                    if not found:
                                        info_dict[f"day{i}_info"][tag].append({
                                            "left_id": left_id, "left_role": left_role,
                                            "right_id": right_id, "right_role": right_role,
                                            "count": 1
                                        })
                                else:
                                    print("error5")
                            else:
                                info_dict[f"day{i}_info"][tag] = [{
                                    "left_id": left_id, "left_role": left_role,
                                    "right_id": right_id, "right_role": right_role,
                                    "count": 1
                                }]
                elif t in category5:
                    #print("category5:", t)
                    t = t.strip()
                    if t in info_dict[f"day{i}_info"]:
                        info_dict[f"day{i}_info"][t][0]["count"] +=1   # タグのみの処理
                    else:
                        info_dict[f"day{i}_info"][t] = [{"count": 1}]
                else:
                    print("カテゴリ外のタグです:", t)
        summarize_list.append(info_dict)
    return summarize_list

import json
import os

def process_csv_files(input_dir, output_dir):
    """Process CSV files and save results as JSON."""
    for target_path in glob.glob(os.path.join(input_dir, "*.csv")):
        # Read CSV
        try:
            df = pd.read_csv(target_path, header=0, encoding="shift_jis")
        except:
            df = pd.read_csv(target_path, header=0, encoding="utf-8")
        
        # Convert date column
        try:
            df["date"] = df["date"].astype(int)
        except Exception as e:
            print("date列の型変換に失敗した行:")
            print(df.loc[df["date"].apply(lambda x: not str(x).isdigit())])
        
        print("path:", target_path)
        
        # Extract execute ID
        try:
            exec_id = int(df[(df["type"] == "execute") & (df["date"] == 1)]["info1"].values[0])
        except (ValueError, IndexError):
            exec_id = None
        
        # Extract attack ID
        try:
            attack_id = int(df[(df["type"] == "attack") & (df["date"] == 1)]["info1"].values[0])
        except (ValueError, IndexError):
            attack_id = None
        
        # Extract divine info
        try:
            recepient_id = df[(df["type"] == "divine") & (df["date"] == 1)]["info2"].values[0]
            result = df[(df["type"] == "divine") & (df["date"] == 1)]["info3"].values[0]
            div_dict = {"agent": recepient_id, "result": result}
        except (ValueError, IndexError):
            div_dict = None
        
        # Aggregate and summarize tags
        dict_list = aggregate_tags(target_path)
        for d in dict_list:
            if d["role"] == "SEER":
                d["Div1_info"] = div_dict
            else:
                d["Divine_info"] = None
        
        result = summarize_tags(dict_list)
        
        # Save as JSON
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.basename(target_path)
        save_file = os.path.join(output_dir, filename.replace(".csv", ".json"))
        result_dict = {"filename": filename, "exec_id": exec_id, "attack_id": attack_id, "agent": result}
        
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        print("result:", result_dict)

# Call the function
#input_directory = r"C:\Users\takic\OneDrive\デスクトップ\修論関係\大会ログ\2025春季\taged_data(特徴量作成用)2"
#output_directory = r"C:\Users\takic\OneDrive\デスクトップ\修論関係\大会ログ\2025春季\2025_spring_json4"
#process_csv_files(input_directory, output_directory)