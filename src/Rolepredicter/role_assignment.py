"""
Role Assignment Functions

This module provides functions to assign roles to players based on model predictions,
considering game constraints such as divination results and execution/attack history.
"""

import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def assign_roles_for_non_seer(
    logits, y_batch, role_counts, role_names, label_encoder,
    fixed_self_role_name, exec_id_batch, attack_id_batch, day2_flag=True, debug=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    自分が指定の役職の場合、他の4人の役職を割り当てる関数。
    
    制約条件：
    - 追放者・被襲撃者は人狼ではない（2日目以降）
    - 指定された役職が自分の視点で最も確率の高い割り当てを選択
    
    Parameters
    ----------
    logits : np.ndarray
        モデルの出力ロジット (5, num_roles)
    y_batch : np.ndarray
        正解の役職ラベル
    role_counts : dict
        各役職の人数
    role_names : list or np.ndarray
        役職名のリスト
    label_encoder : LabelEncoder
        役職のラベルエンコーダー
    fixed_self_role_name : str
        自分の役職名
    exec_id_batch : np.ndarray
        追放者ID
    attack_id_batch : np.ndarray
        被襲撃者ID
    day2_flag : bool, default=True
        2日目以降のルールを適用するか
    debug : bool, default=False
        デバッグ出力を表示するか
        
    Returns
    -------
    tuple
        (予測役職 (n_samples, 4), 正解役職 (n_samples, 4))
    """
    
    role_names = list(role_names)
    num_players = logits.shape[0]
    if num_players != 5:
        return np.array([]), np.array([])
    
    pred_list, y_list = [], []

    # 各プレイヤーを「自分」と仮定してループ
    for self_index in range(num_players):
        self_role = y_batch[self_index]
        self_role_name = label_encoder.inverse_transform([self_role])[0]
        
        # 実際の役職が指定の役職でない場合はスキップ
        if self_role_name != fixed_self_role_name:
            continue

        # 追放者・被襲撃者の場合はスキップ
        if day2_flag and (self_index + 1 in exec_id_batch or self_index + 1 in attack_id_batch):
            if debug:
                print(f"skip id:{self_index + 1} exec or attack")
            continue

        other_indices = [i for i in range(num_players) if i != self_index]
        logits_others = logits[other_indices]
        y_others = y_batch[other_indices]
        
        # 制約：追放者・被襲撃者は人狼ではない
        non_werewolf_indices_in_others = []
        if day2_flag:
            exec_id = exec_id_batch[self_index]
            attack_id = attack_id_batch[self_index]
            if np.isnan(attack_id) or np.isnan(exec_id): 
                continue
            attack_id_idx = int(attack_id) - 1
            exec_id_idx = int(exec_id) - 1
            if self_index == exec_id_idx or self_index == attack_id_idx: 
                continue
            if exec_id_idx in other_indices:
                non_werewolf_indices_in_others.append(other_indices.index(exec_id_idx))
            if attack_id_idx in other_indices:
                non_werewolf_indices_in_others.append(other_indices.index(attack_id_idx))
            non_werewolf_indices_in_others = list(set(non_werewolf_indices_in_others))
            
        # 役職プールを作成
        reduced_counts = role_counts.copy()
        if reduced_counts.get(fixed_self_role_name, 0) == 0:
            continue
        reduced_counts[fixed_self_role_name] -= 1
        roles_pool = [role for role, count in reduced_counts.items() for _ in range(count)]
        if len(roles_pool) != 4: 
            continue
            
        # 最適な役職割り当てを探索
        best_perm = None
        best_score = -np.inf
        for perm in set(itertools.permutations(roles_pool)):
            # 人狼制約をチェック
            if any(perm[idx] == "WEREWOLF" for idx in non_werewolf_indices_in_others):
                continue
            # スコア計算（対数尤度）
            score = sum(np.log(logits_others[i][role_names.index(role)] + 1e-9) for i, role in enumerate(perm))
            if score > best_score:
                best_score, best_perm = score, perm
                
        # 最適な割り当てを結果に追加
        if best_perm:
            pred_encoded = label_encoder.transform(list(best_perm))
            pred_list.append(pred_encoded)
            y_list.append(y_others)

    return (np.concatenate(pred_list) if pred_list else np.array([]),
            np.concatenate(y_list) if y_list else np.array([]))


def assign_roles_for_seer_by_divination(
    logits, y_batch, role_counts, role_names, label_encoder,
    div_result_array1, div_id_array1, div_result_array2=None, div_id_array2=None,
    exec_id_batch=None, attack_id_batch=None, day2_flag=True, debug=False) -> dict:
    """
    占い師視点での役職割り当て関数。占い結果を制約として使用。
    
    制約条件：
    - 追放者・被襲撃者は人狼ではない
    - 占い結果に基づく役職判定（黒 = 人狼、白 = 人狼ではない）
    - 1日目の占い結果に基づき、black/white に分類
    
    Parameters
    ----------
    logits : np.ndarray
        モデルの出力ロジット (5, num_roles)
    y_batch : np.ndarray
        正解の役職ラベル
    role_counts : dict
        各役職の人数
    role_names : list or np.ndarray
        役職名のリスト
    label_encoder : LabelEncoder
        役職のラベルエンコーダー
    div_result_array1 : np.ndarray
        1日目の占い結果
    div_id_array1 : np.ndarray
        1日目の占い対象ID
    div_result_array2 : np.ndarray, optional
        2日目の占い結果
    div_id_array2 : np.ndarray, optional
        2日目の占い対象ID
    exec_id_batch : np.ndarray, optional
        追放者ID
    attack_id_batch : np.ndarray, optional
        被襲撃者ID
    day2_flag : bool, default=True
        2日目以降のルールを適用するか
    debug : bool, default=False
        デバッグ出力を表示するか
        
    Returns
    -------
    dict
        {
            'black': (予測役職, 正解役職),
            'white': (予測役職, 正解役職)
        }
    """
    
    role_names = list(role_names)
    
    num_players = logits.shape[0]
    if num_players != 5:
        empty_result = (np.array([]), np.array([]))
        return {"black": empty_result, "white": empty_result}

    pred_list_black, y_list_black = [], []
    pred_list_white, y_list_white = [], []

    fixed_self_role_name = "SEER"
    werewolf_label = "WEREWOLF"

    # 各プレイヤーを「自分(占い師)」と仮定してループ
    for self_index in range(num_players):
        self_role_name = label_encoder.inverse_transform([y_batch[self_index]])[0]

        # 実際の役職が占い師でない場合はスキップ
        if self_role_name != fixed_self_role_name:
            continue
        
        # 追放者・被襲撃者の場合はスキップ
        if day2_flag and (self_index + 1 in exec_id_batch or self_index + 1 in attack_id_batch):
            continue

        if div_id_array1 is None or all(np.isnan(div_id_array1)):
            continue

        other_indices = [i for i in range(num_players) if i != self_index]
        logits_others = logits[other_indices]
        y_others = y_batch[other_indices]

        # 制約の初期化
        non_werewolf_indices_in_others = []
        div_constraints = {}  # {target_idx_in_others: "WEREWOLF" or "HUMAN"}

        # 制約1：追放者・被襲撃者は人狼ではない
        if day2_flag and exec_id_batch is not None and attack_id_batch is not None:
            exec_id = exec_id_batch[self_index]
            attack_id = attack_id_batch[self_index]
            if not (np.isnan(exec_id) or np.isnan(attack_id)):
                exec_id_idx = int(exec_id) - 1
                attack_id_idx = int(attack_id) - 1
            else:
                continue
            if exec_id_idx in other_indices:
                non_werewolf_indices_in_others.append(other_indices.index(exec_id_idx))
            if attack_id_idx in other_indices:
                non_werewolf_indices_in_others.append(other_indices.index(attack_id_idx))
            non_werewolf_indices_in_others = list(set(non_werewolf_indices_in_others))
        
        # 制約2：占い結果を収集
        all_div_results = [(div_result_array1, div_id_array1)]
        if day2_flag and div_result_array2 is not None and div_id_array2 is not None:
            all_div_results.append((div_result_array2, div_id_array2))

        first_div_result_for_classification = None

        for i, (div_result_array, div_id_array) in enumerate(all_div_results):
            if np.isnan(div_id_array[self_index]): 
                continue
            
            div_id = int(div_id_array[self_index]) - 1
            if div_id not in other_indices: 
                continue
                
            div_target_index_in_others = other_indices.index(div_id)
            div_result_str = str(div_result_array[self_index]).strip()

            # 占い結果を解析
            div_logic_result = None
            if div_result_str in ["WEREWOLF", "人狼", "黒"]:
                div_logic_result = "WEREWOLF"
            elif div_result_str in ["HUMAN", "白", "not(人狼)"]:
                div_logic_result = "HUMAN"
            
            if div_logic_result:
                div_constraints[div_target_index_in_others] = div_logic_result
                if i == 0:  # 1日目の占い結果を分類用に保存
                    first_div_result_for_classification = div_logic_result
        
        # 1日目の占い結果がなければスキップ
        if first_div_result_for_classification is None:
            continue

        # 役職プールを作成
        reduced_counts = role_counts.copy()
        if reduced_counts.get(fixed_self_role_name, 0) == 0: 
            continue
        reduced_counts[fixed_self_role_name] -= 1
        roles_pool = [role for role, count in reduced_counts.items() for _ in range(count)]
        if len(roles_pool) != 4: 
            continue

        # 最適な役職割り当てを探索
        best_score, best_perm = -np.inf, None

        for perm in set(itertools.permutations(roles_pool)):
            is_valid = True
            
            # チェック1：追放者・被襲撃者制約
            if any(perm[idx] == werewolf_label for idx in non_werewolf_indices_in_others):
                continue
                
            # チェック2：占い結果制約
            for target_idx, result in div_constraints.items():
                if result == "WEREWOLF" and perm[target_idx] != werewolf_label:
                    is_valid = False
                    break
                if result == "HUMAN" and perm[target_idx] == werewolf_label:
                    is_valid = False
                    break
            if not is_valid: 
                continue

            # スコア計算
            score = sum(np.log(logits_others[i][role_names.index(role)] + 1e-9) for i, role in enumerate(perm))
            if score > best_score:
                best_score, best_perm = score, perm

        # 1日目の占い結果に基づき分類
        if best_perm:
            pred_encoded = label_encoder.transform(list(best_perm))
            if first_div_result_for_classification == "WEREWOLF":
                pred_list_black.append(pred_encoded)
                y_list_black.append(y_others)
            elif first_div_result_for_classification == "HUMAN":
                pred_list_white.append(pred_encoded)
                y_list_white.append(y_others)

    result = {
        "black": (np.concatenate(pred_list_black) if pred_list_black else np.array([]),
                  np.concatenate(y_list_black) if y_list_black else np.array([])),
        "white": (np.concatenate(pred_list_white) if pred_list_white else np.array([]),
                  np.concatenate(y_list_white) if y_list_white else np.array([])),
    }
    return result
