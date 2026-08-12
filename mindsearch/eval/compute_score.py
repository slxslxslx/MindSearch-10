import json
import re
import string


def read_jsonl_file(file_path):
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f: # 以UTF-8编码打开文件，避免中文乱码
        for line_num, line in enumerate(f, 1):
            line = line.strip() 
            if not line: 
                continue
            try:
                # 逐行解析JSON
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                # 捕获解析错误，提示具体行号便于排查
                print(f"第{line_num}行JSON解析失败: {e}")
                # 可选：遇到错误是否继续读取，这里选择继续
                continue
    return data


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



def compute_score(file, out_path=None):

    data = read_jsonl_file(file)

    correct = 0
    scores = []

    for item in data:
        print(f'第{item["id"]}条数据的decision是：{item["judge"]["decision"]}')
        if item["judge"]["decision"]:
            correct += 1

        print(f'第{item["id"]}条数据的decision是：{item["short_answer"]}')
        gold = item.get("answer", "")
        short_answer = item.get("short_answer", "")
        scores.append(int(exact_match(short_answer, gold)))

    total = len(data)
    acc = correct / total
    print("Accuracy:", acc)

    em_score = sum(scores) / len(scores)

    # 保存结果到 out_path
    if out_path:
        result = {"accuracy": round(acc, 4), "acc_correct_count": correct, "total": total, "EM": em_score, "EM_correct_count": sum(scores)}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {out_path}")

    return acc, correct, total


# 示例调用
if __name__ == "__main__":
    # file_path = "mindsearch/results/bamboogle_data_judged-1.jsonl"
    # output_path = "mindsearch/results/accuracy_result.json"
    # compute_accuracy(file_path, output_path)


    # file_path = "mindsearch/results/bamboogle_data_googleP_judged-1.jsonl"
    # output_path = "mindsearch/results/accuracy_googleP_result.json"

    # file_path = "mindsearch/results/bamboogle_data_tencent_12P_judged-1.jsonl"
    # output_path = "mindsearch/results/accuracy_tencent_12P_result.json"


    file_path = "mindsearch/results/bamboogle_data_tencent_12P_tempStopMaxTMaxTurn_judged-1.jsonl"
    output_path = "mindsearch/results/accuracy_tencent_12P_tempStopMaxTMaxTurn_result.json"

    compute_score(file_path, output_path)


    # file_path = "mindsearch/results/bamboogle_data_judged-1.jsonl"
    # jsonl_data = read_jsonl_file(file_path)
    
    # print(f"共读取到 {len(jsonl_data)} 条数据")

    # if jsonl_data:
    #     print("第一条数据：", jsonl_data[0])



# def compute_accuracy(file, out_path=None):

#     data = read_jsonl_file(file)

#     correct = 0

#     for item in data:
#         print(f'第{item["id"]}条数据的decision是：{item["judge"]["decision"]}')
#         if item["judge"]["decision"]:
#             correct += 1

#     total = len(data)
#     acc = correct / total

#     print("Accuracy:", acc)

#     # 保存结果到 out_path
#     if out_path:
#         result = {
#             "accuracy": round(acc, 4),
#             "correct": correct,
#             "total": total
#         }
#         with open(out_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, ensure_ascii=False, indent=2)
#         print(f"结果已保存到: {out_path}")

#     return acc, correct, total

