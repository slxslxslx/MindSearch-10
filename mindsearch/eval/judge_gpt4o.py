# 在文件最顶部添加（第1-3行）
from dotenv import load_dotenv
import os
load_dotenv()  # 自动加载 .env 文件

import openai
import json
from tqdm import tqdm
import re

JUDGE_PROMPT = """
Task
I need your help in evaluating an answer provided by an LLM against a ground truth answer for a given question. Your task is to determine if the ground truth answer is present in the LLM's response. Please analyze the provided data and make a decision.

Instructions
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer."
2. Consider the substance of the answers - look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answer" are present in the "Predicted Answer."

Input Data
- Question: {question}
- Predicted Answer: {prediction}
- Ground Truth Answer: {answer}

Output Format
You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads.
Return ONLY valid JSON.
Do NOT wrap the JSON in markdown or code fences.
Do NOT add any other text before or after the JSON.

Response Format:
{{
"Explanation": "(How you made the decision?)",
"Decision": "TRUE" or "FALSE"
}}

Your output:
"""




def extract_json(text):

    text = text.strip()

    # 去掉 ```json
    if text.startswith("```"):
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = text.strip("` \n")

    # 提取 { ... }
    match = re.search(r"\{.*\}", text, re.S)

    if match:
        text = match.group(0)

    return json.loads(text)

def judge(question, answer, prediction, model_name="gpt-4o-free"):

    if isinstance(answer, str):
        answer = [answer]

    prompt = JUDGE_PROMPT.format(
        question = question,
        answer = json.dumps(answer, ensure_ascii=False),
        prediction = prediction
    )
    client = openai.OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),  # my
        base_url = "https://aihubmix.com/v1"
    )

    response = client.chat.completions.create(
        model=model_name, # "gpt-4o-free",
        #   model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    text = response.choices[0].message.content
    print("################text:",text)
    # ################text: {
    # "Explanation": "The Predicted Answer states that James Madison was the President of the United States during the year Citibank was founded (1812). The Ground Truth Answer also identifies James Madison as the President. Though the wording differs, the critical information, 'James Madison,' is present in the Predicted Answer.",
    # "Decision": "TRUE"
    # }


    try:
        result = extract_json(text)# json.loads(text)
        print("########################result:", result)
        decision = result["Decision"].lower() == "true"
        return decision, result, text

    except Exception:
        return False, {"error": "invalid_json"}, text


def run_judge(prediction_file, output_file):
    import time
    model_name= "gpt-4o" # "gpt-4o-free"
    print("Start judging...")

    # ============ 🔑 新增：智能跳过已处理样本 ============
    # 1. 读取已存在的判断结果（避免重复调用API）
    existing_ids = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f_exist:
            for line in f_exist:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # 统一转为字符串比较（兼容int/str类型的id）
                        existing_ids.add(str(data.get("id", "")).strip())
                    except (json.JSONDecodeError, AttributeError):
                        continue  # 跳过无效行
        if existing_ids:
            print(f"✅ 已检测到 {len(existing_ids)} 个已判断样本，将自动跳过")
    except FileNotFoundError:
        print("📝 首次运行，将创建新结果文件")

    # 2. 用户指定跳过ID（保留原逻辑，统一转为字符串）
    user_skip_ids = {str(x) for x in []}  # 如需调整请修改此处
    skip_set =  existing_ids | user_skip_ids
    # ============ 🔑 新增结束 ============
  
    processed_count = 0
    skipped_count = 0

    with open(prediction_file, "r", encoding="utf-8") as fin, \
         open(output_file, "a", encoding="utf-8") as fout:

        total_lines = sum(1 for _ in fin)  # 预计算总数（用于进度条）
        fin.seek(0)  # 重置文件指针

        for line in tqdm(fin, total=total_lines, desc="Judging samples"):
            line = line.strip()

            if not line:
                continue

            sample = json.loads(line)
            sample_id  = sample.get("id", "N/A")  # 取 id，没有就显示 N/A

            sample_id_str = str(sample_id).strip()

            # ============ 🚦 智能跳过逻辑 ============
            if sample_id_str in skip_set:
                skipped_count += 1
                if sample_id_str in user_skip_ids:
                    print(f"⏭️  跳过用户指定样本: id={sample_id} (在跳过列表中)")
                else:
                    print(f"⏭️  跳过已判断样本: id={sample_id} (结果已存在)")
                continue
            # ============ 🚦 跳过逻辑结束 ============

            question = sample["question"]
            answer = sample["answer"]
            prediction = sample["prediction"]

            print(f"\n🔍 正在处理新样本: id={sample_id} (剩余API调用: {total_lines - processed_count - skipped_count})")


            
            # ============ ⚠️ 安全调用judge（带错误重试） ============
            max_retries = 7
            for attempt in range(5,max_retries):
                try:
                    time.sleep(40)  # ✅ 安全！每个新样本调用前固定等待3秒
                    decision, result, raw_text = judge(
                        question, answer, prediction, model_name
                    )
                    break  # 成功则跳出重试循环
                except Exception as e:
                    print('$$$$$$$$$$$$$$$$$$$$$e:',e)
                    if "RateLimitError" in str(type(e).__name__) or "429" in str(e):
                        wait_seconds  = 2 ** attempt  # 指数退避
                        print(f"⏳ 遇到API限流，{wait_seconds }秒后重试 (第{attempt+1}/{max_retries}次)...")
                        time.sleep(wait_seconds )
                        if attempt == max_retries - 1:
                            print(f"❌ 样本 id={sample_id} 判断失败（API配额耗尽），已保存进度")
                            # 保存当前进度后退出（避免浪费剩余配额）
                            fout.flush()
                            return
                    else:
                        raise  # 非限流错误直接抛出
            # ============ ⚠️ 调用结束 ============
            print("################result:",result)

            judged_sample = {
                "id":sample_id,
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "judge": {
                    "decision": decision,
                    "explanation": result.get("Explanation"),
                    "raw_output": raw_text, 
                    "judge_model": model_name
                }
            }

            fout.write(json.dumps(judged_sample, ensure_ascii=False) + "\n" )
            fout.flush()

            processed_count += 1
            skip_set.add(sample_id_str)  # 动态更新已处理集合（防重复）

    print(f"\n✅ 判断完成! 新增: {processed_count} 条 | 跳过: {skipped_count} 条")
    print(f"💡 提示: 再次运行将自动跳过所有已处理样本，节省API配额")

if __name__ == "__main__":
    input_path  = "/home/shaolingxuan/project/slx-MindSearch/mindsearch/results/A-GoogleSearch_qwen-3.5-9b-think-tool_seal0/all.jsonl"
    output_path = "/home/shaolingxuan/project/slx-MindSearch/mindsearch/results/A-GoogleSearch_qwen-3.5-9b-think-tool_seal0/4o-judge.jsonl"
    run_judge(input_path, output_path)
