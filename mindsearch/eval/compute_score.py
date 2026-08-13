import json
import re
import string
from collections import Counter  # 新增：用于计算词频，实现标准的Token-level F1

def read_jsonl_file(file_path):
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:  # 以UTF-8编码打开文件，避免中文乱码
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                # 逐行解析JSON
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析失败: {e}")
                continue
    return data


# https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py
def normalize_answer(s):
    """小写、去标点、去冠词、去多余空格"""

    def remove_articles(text):  # 移除冠词 a/an/the
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):  # 清理多余空格
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)  # 获取所有标点符号：!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# def f1_score(prediction, ground_truth):
#     """
#     计算Token级别的F1分数 (SQuAD标准)
#     """
#     pred_tokens = normalize_answer(prediction).split()
#     gold_tokens = normalize_answer(ground_truth).split()
    
#     # 如果预测和标准答案都为空，认为完全匹配，F1为1
#     if len(pred_tokens) == 0 and len(gold_tokens) == 0:
#         return 1.0
#     # 如果其中一个为空，则没有重叠，F1为0
#     if len(pred_tokens) == 0 or len(gold_tokens) == 0:
#         return 0.0
    
#     # 使用Counter计算交集（考虑同一个词出现的次数）
#     common = Counter(pred_tokens) & Counter(gold_tokens)
#     num_same = sum(common.values())
    
#     if num_same == 0:
#         return 0.0
    
#     precision = 1.0 * num_same / len(pred_tokens)
#     recall = 1.0 * num_same / len(gold_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
    
#     return f1


# 新增/修改：同时计算并返回 Precision, Recall, F1
def compute_token_metrics(prediction, ground_truth):
    """
    计算Token级别的 Precision, Recall, F1 分数 (SQuAD标准)
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    # 如果预测和标准答案都为空，认为完全匹配
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0, 1.0, 1.0
    # 如果其中一个为空，则没有重叠
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0, 0.0, 0.0
    
    # 使用Counter计算交集（考虑同一个词出现的次数）
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0, 0.0, 0.0
    
    # Precision: 生成的答案中有多少是正确的
    precision = 1.0 * num_same / len(pred_tokens)
    # Recall: 标准答案中有多少被成功生成了
    recall = 1.0 * num_same / len(gold_tokens)
    # F1: 综合两者的调和平均数
    f1 = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1

def compute_score(file, out_path=None):

    data = read_jsonl_file(file)

    correct = 0
    em_scores  = []
    f1_scores = []  # 新增：用于存储每条数据的F1分数
    precision_scores = [] # 新增：用于存储 Precision
    recall_scores = []    # 新增：用于存储 Recall

    for item in data:
        print(f'第{item["id"]}条数据的decision是：{item["judge"]["decision"]}')
        if item["judge"]["decision"]:
            correct += 1

        print(f'第{item["id"]}条数据的short_pred是：{item["short_pred"]}')
        gold = item.get("answer", "")
        short_pred = item.get("short_pred", "")

        # 计算EM
        em_scores.append(int(exact_match(short_pred, gold)))

        # # 计算F1
        # f1_scores.append(f1_score(short_pred, gold))

        # 计算P, R, F1
        p, r, f1 = compute_token_metrics(short_pred, gold)
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f1)

    total = len(data)
    acc = correct / total
    print("Accuracy: %.2f%%" % (acc * 100))

    em_score = sum(em_scores) / len(em_scores) if len(em_scores) > 0 else 0
    print("EM: %.2f%%" % (em_score * 100))

    # # 新增：计算平均F1
    # avg_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0
    # print("F1:", avg_f1)

    avg_precision = sum(precision_scores) / len(precision_scores) if len(precision_scores) > 0 else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if len(recall_scores) > 0 else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0
    
    print("Precision: %.2f%%" % (avg_precision * 100))
    print("Recall: %.2f%%" % (avg_recall * 100))
    print("F1: %.2f%%" % (avg_f1 * 100))

    # 保存结果到 out_path
    if out_path:
        result = {
            "accuracy": "%.2f%%" % (acc * 100),
            "acc_correct_count": correct,
            "total": total,
            "EM": "%.2f%%" % (em_score * 100),
            "EM_correct_count": sum(em_scores),
            "Precision": "%.2f%%" % (avg_precision * 100), # 新增保存 Precision
            "Recall": "%.2f%%" % (avg_recall * 100),       # 新增保存 Recall
            "F1": "%.2f%%" % (avg_f1 * 100)
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {out_path}")

    return acc, correct, total, em_score, avg_f1


# 示例调用
if __name__ == "__main__":
    # file_path = "mindsearch/results/bamboogle_data_judged-1.jsonl"
    # output_path = "mindsearch/results/accuracy_result.json"
    # compute_accuracy(file_path, output_path)

    # file_path = "mindsearch/results/bamboogle_data_googleP_judged-1.jsonl"
    # output_path = "mindsearch/results/accuracy_googleP_result.json"

    # file_path = "mindsearch/results/bamboogle_data_tencent_12P_judged-1.jsonl"
    # output_path = "mindsearch/results/accuracy_tencent_12P_result.json"

    # file_path = "results/Serpbing-judge-bamboogle/bamboogle_SerpBing-juged.jsonl"
    # output_path = "results/accuracy_bing_result.json"

    # file_path = "results/SerpBingSearch_qwen-3.5-9b_bamboogle/search3次重试+select打分+searchQ优化/juged.jsonl"
    # output_path = "results/accuracy_bing_result.json"
    
    # file_path = "results/SerpBingSearch_qwen-3.5-9b_bamboogle/3.5适配-base/juged.jsonl"
    # output_path = "results/accuracy_bing_result.json"

    # file_path = "results/accuracy_google-bamboogle-wiki_chunk_juged.jsonl"
    # output_path = "results/accuracy_google-bamboogle-wiki_chunk_score.json"


    file_path = "/home/shaolingxuan/project/MindSearch-10/mindsearch/results/GoogleSearch-internlm_7b_chat-seal0/4o-judge.jsonl"
    output_path = "/home/shaolingxuan/project/MindSearch-10/mindsearch/results/GoogleSearch-internlm_7b_chat-seal0/score.jsonl"

    compute_score(file_path, output_path)

    # file_path = "mindsearch/results/bamboogle_data_judged-1.jsonl"
    # jsonl_data = read_jsonl_file(file_path)

    # print(f"共读取到 {len(jsonl_data)} 条数据")

    # if jsonl_data:
    #     print("第一条数据：", jsonl_data[0])