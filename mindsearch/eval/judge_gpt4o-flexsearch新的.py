# 在文件最顶部添加（第1-3行）
import os

from dotenv import load_dotenv

load_dotenv()  # 自动加载 .env 文件

import json
import re

import openai
from tqdm import tqdm


#@@@@@@@@@@@@@@@@@@
# 从模型输出中提取短答案
def extract_short_answer(prediction: str, question: str = None) -> str:
    """
    从模型的长回答中提取短答案，风格与 WebThinker 的 extract_answer_fn 一致。
    优先匹配 \\boxed{...}，其次匹配 ANSWER: / Final Answer: 等标记。
    """
    prediction = prediction or ""
    # 1. 优先匹配 \\boxed{...}
    boxed_pattern = r"\\boxed\{(.*)\}"
    boxed_matches = re.findall(boxed_pattern, prediction, re.S)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # 2. 其次匹配 Final Answer: / ANSWER: 等
    for ans_pattern in ["Final Answer:", "final answer:", "FINAL ANSWER:",
                        "ANSWER:", "Answer:"]:
        if ans_pattern in prediction:
            extracted_text = prediction.split(ans_pattern)[-1].strip('**').strip().strip('.')
            # 去掉可能存在的 \\text{...}
            inner_pattern = r"\\text\{(.*)\}"
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1].strip("()")
            return extracted_text.strip()

    # 3. 都没有，取最后 3 行
    lines = prediction.replace("\n\n", "\n").strip().split('\n')
    last_lines = "\n".join(lines[-3:]).strip()
    return last_lines



#@@@@@@@@@@@@@@@@@@
JUDGE_PROMPT = """
## Task
You are an evaluation assistant. Given a [Question], a long-form [Response] produced by an AI agent, and a [Ground Truth Answer], your task is to:
  (1) Extract the final, concise answer from the [Response].
  (2) Judge whether the extracted answer is equivalent to the [Ground Truth Answer].

## Instructions
### Step 1 — Extract the final answer
  - Carefully read the [Response] and identify the final answer the agent provides to the [Question].
  - Extract only the core answer (e.g., a number, a name, a short phrase, or a choice like "B").
  - Put the extracted answer as a string. If no clear final answer can be found in the [Response], put "None".

### Step 2 — Compare with the ground truth
  - Compare the substance of the extracted answer with the [Ground Truth Answer].
  - Look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning.
  - For numerical problems, accept answers within a small margin of error.
  - Your final decision should be based on whether the meaning and vital facts of the [Ground Truth Answer] are present in the extracted answer.

## Input Data
- Question: {question}
- Response: {response}
- Ground Truth Answer: {answer}

## Output Format
You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads.
Return ONLY valid JSON.
Do NOT wrap the JSON in markdown or code fences.
Do NOT add any other text before or after the JSON.

## Response Format:
{{
"Extracted_Answer": "(The final concise answer extracted from the Response. Put 'None' if no clear final answer is found.)",
"Explanation": "(How you extracted the answer and why it is equivalent or not equivalent to the Ground Truth Answer. Focus only on whether the answers match; do not solve the problem independently or argue for a different answer.)",
"Decision": "TRUE" or "FALSE"
}}

Your output:
"""

JUDGE_PROMPT_ZH = """
## 任务
你是一名评测助手。给定【问题】、AI智能体生成的长篇【回答文本】以及【标准答案（真实参考值）】，你的工作分为两步：
  (1) 从【回答文本】中提取简洁的最终答案。
  (2) 判断提取出的答案是否与【标准答案】语义等价。

## 操作指引
### 第一步 — 提取最终答案
  - 仔细阅读【回答文本】，定位智能体针对【问题】给出的最终答案。
  - 仅提取核心答案（例如：数字、人名、短语、选择题选项如“B”）。
  - 将提取结果以字符串形式填写。若回答文本中找不到清晰明确的最终答案，则填写 "None"。

### 第二步 — 与标准答案比对
  - 对比提取答案与【标准答案】的核心实质内容。
  - 重点判断信息是否等价、答案是否正确；不必苛求文字完全一模一样，除非文字本身是决定语义的关键。
  - 数值类问题：答案误差在极小合理范围内即判定为正确。
  - 最终判定依据：提取答案是否包含标准答案的核心含义与关键事实。

## 输入数据
- 问题：{question}
- 回答文本：{response}
- 标准答案：{answer}

## 输出格式要求
你只能按照下述规范输出JSON格式内容，保证输出内容可被Python的json.loads函数正常解析。
仅输出合规JSON，禁止输出其他内容。
禁止使用Markdown标记、代码块符号包裹JSON文本。
禁止在JSON前后添加任何额外文字、注释。

## 输出JSON结构：
{
"Extracted_Answer": "此处填写从回答文本提取的简洁最终答案，无明确答案则填'None'",
"Explanation": "说明答案提取逻辑，以及该答案与标准答案等价/不等价的原因。仅围绕两份答案是否匹配展开说明；禁止自行求解题目、禁止论证其他答案的合理性。",
"Decision": "TRUE" 或 "FALSE"
}

你的输出：
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

    # 1. 先从模型长输出中提取短答案   #@@@@@@@@@@@@@@@@@@
    # short_pred = extract_short_answer(prediction, question)
    # print(f"q={question}。。。。short_pred={short_pred}")

    # 2. 构造 LLM-judge prompt
    prompt = JUDGE_PROMPT.format(question=question, answer=json.dumps(answer, ensure_ascii=False), response=prediction)  #@@@@@@@@@@@@@@@@@@
    # prompt = JUDGE_PROMPT_ZH.format(question=question, answer=json.dumps(answer, ensure_ascii=False), response=response)  #@@@@@@@@@@@@@@@@@@
    
    # print("##########prompt：",prompt)

    # response = openai.ChatCompletion.create(
    #     model="gpt-4o-free",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0
    # )
    # text = response["choices"][0]["message"]["content"]

    # print(os.getenv("OPENAI_API_KEY"))

    # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.inferera.com")  # my  https://aihubmix.com/v1
   
   # 3. 调用模型
    import httpx  # 新增：导入 httpx
    # ================= 新增：配置 SOCKS5 代理 =================
    proxy_url = "socks5h://127.0.0.1:1824"
    # 创建带有代理的 httpx 客户端
    http_client = httpx.Client(proxy=proxy_url)

    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url="https://aihubmix.com/v1" ,
        http_client=http_client  # 传入自定义的 httpx 客户端
    )  
    # =========================================================

    response = client.chat.completions.create(
        model=model_name,  # "gpt-4o-free",
        #   model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    print("################text:", text)
    # ################text: {
    # "Explanation": "The Predicted Answer states that James Madison was the President of the United States during the year Citibank was founded (1812). The Ground Truth Answer also identifies James Madison as the President. Though the wording differs, the critical information, 'James Madison,' is present in the Predicted Answer.",
    # "Decision": "TRUE"
    # }

    # 4. 解析 LLM-judge 输出
    try:
        result = extract_json(text)  # json.loads(text)
        print("########################result:", result)
        decision = result["Decision"].lower() == "true"
        return decision, result, text

    except Exception:
        return False, {"error": "invalid_json"}, text


def run_judge(prediction_file, output_file):
    import time

    model_name = "gpt-4o"  # "gpt-4o-free" # gpt-4o
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
    skip_set = existing_ids | user_skip_ids
    # ============ 🔑 新增结束 ============

    processed_count = 0
    skipped_count = 0

    with open(prediction_file, "r", encoding="utf-8") as fin, open(output_file, "a", encoding="utf-8") as fout:

        total_lines = sum(1 for _ in fin)  # 预计算总数（用于进度条）
        fin.seek(0)  # 重置文件指针

        for line in tqdm(fin, total=total_lines, desc="Judging samples"):
            line = line.strip()

            if not line:
                continue

            sample = json.loads(line)
            sample_id = sample.get("id", "N/A")  # 取 id，没有就显示 N/A

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
            # short_answer = sample["short_answer"]

            print(f"\n🔍 正在处理新样本: id={sample_id} (剩余API调用: {total_lines - processed_count - skipped_count})")

            # if sample_id in [1,2,3,4]:
            #     continue

            # decision, result, raw_text = judge(
            #     question,
            #     answer,
            #     prediction,
            #     model_name
            # )

            # ============ ⚠️ 安全调用judge（带错误重试） ============
            max_retries = 7
            for attempt in range(5, max_retries):
                try:
                    time.sleep(5) # 20 # ✅ 安全！每个新样本调用前固定等待3秒
                    decision, result, raw_text = judge(question, answer, prediction, model_name)
                    break  # 成功则跳出重试循环
                except Exception as e:
                    print('$$$$$$$$$$$$$$$$$$$$$e:', e)
                    if "RateLimitError" in str(type(e).__name__) or "429" in str(e):
                        wait_seconds = 2**attempt  # 指数退避
                        print(f"⏳ 遇到API限流，{wait_seconds }秒后重试 (第{attempt+1}/{max_retries}次)...")
                        time.sleep(wait_seconds)
                        if attempt == max_retries - 1:
                            print(f"❌ 样本 id={sample_id} 判断失败（API配额耗尽），已保存进度")
                            # 保存当前进度后退出（避免浪费剩余配额）
                            fout.flush()
                            return
                    else:
                        raise  # 非限流错误直接抛出
            # ============ ⚠️ 调用结束 ============
            print("################result:", result)

            judged_sample = {"id": sample_id, "question": question, "answer": answer, "prediction": prediction,  "judge": {"decision": decision,  "Extracted_Answer": result.get("Extracted_Answer"), "explanation": result.get("Explanation"), "raw_output": raw_text, "judge_model": model_name}}

            fout.write(json.dumps(judged_sample, ensure_ascii=False) + "\n")
            fout.flush()

            processed_count += 1
            skip_set.add(sample_id_str)  # 动态更新已处理集合（防重复）

    print(f"\n✅ 判断完成! 新增: {processed_count} 条 | 跳过: {skipped_count} 条")
    print(f"💡 提示: 再次运行将自动跳过所有已处理样本，节省API配额")


if __name__ == "__main__":

    # input_path = "results/SerpbingP-bamboogle/Predict-all.jsonl"
    # input_path = "results/SerpBingSearch_qwen-3.5-9b_bamboogle/search3次重试+select打分+searchQ优化/Predict-all.jsonl"
    # output_path = "results/SerpBingSearch_qwen-3.5-9b_bamboogle/search3次重试+select打分+searchQ优化/juged.jsonl"
    
    
    # input_path = "results/SerpBingSearch_qwen-3.5-9b_bamboogle/3.5适配-base/Predict-all.jsonl"
    # output_path = "results/SerpBingSearch_qwen-3.5-9b_bamboogle/3.5适配-base/juged.jsonl"

    # input_path = "results/GoogleSearch_qwen-3.5-9b_seal0/测试长wiki添加chunk/GoogleSearch_qwen-3.5-9b_seal0-all.jsonl"
    # output_path = "results/accuracy_google-seal0-wiki_chunk_juged.jsonl"

    input_path = "/home/shaolingxuan/project/slx-MindSearch/mindsearch/results/A-GoogleSearch_qwen-3.5-9b-think-tool_bamboogle/117-GoogleSearch_qwen-3.5-9b_bamboogle_20260729_162315.jsonl"
    output_path = "/home/shaolingxuan/project/slx-MindSearch/mindsearch/results/A-GoogleSearch_qwen-3.5-9b-think-tool_bamboogle/117-GoogleSearch_qwen-3.5-9b_bamboogle_20260729_162315-4ojudge.jsonl"

    run_judge(input_path, output_path)


# if __name__ == "__main__":
# question = "Who was president of the United States in the year that Citibank was founded?"
# answer = "james madison"
# prediction = "The President of the United States in the year that Citibank was founded (1812) was James Madison. Madison served as the fourth President of the United States from March 4, 1809, to March 4, 1817. He played a crucial role in the early development of the United States, particularly during the War of 1812, which was fought against Great Britain [[1]][[2]]."
# res = judge(question, answer, prediction, "gpt-4o-free")
# print('########res:',res)
# # ########res: (True, {'Explanation': "The Predicted Answer states that James Madison was the President of the United States during the year Citibank was founded (1812). The Ground Truth Answer also identifies James Madison as the President. Though the wording differs, the critical information, 'James Madison,' is present in the Predicted Answer.", 'Decision': 'TRUE'})

# file_path = "mindsearch/results/bamboogle_data_predictions-test.jsonl"
# jsonl_data = read_jsonl_file(file_path)

# print(f"共读取到 {len(jsonl_data)} 条数据")

# if jsonl_data:
#     print("第一条数据：", jsonl_data[0])


# (mindsearch) root@autodl-container-bdf4448313-9cfad8a0:~/autodl-tmp/slx-MindSearch# python -m mindsearch.eval.judge_gpt4o
# Start judging...
# 0it [00:00, ?it/s]################text: ```json
# {
#   "Explanation": "The Predicted Answer correctly identifies James Madison as the President of the United States in 1812, which matches the Ground Truth Answer. The substance and meaning of the Ground Truth Answer are present in the Predicted Answer.",
#   "Decision": "TRUE"
# }
# ```
# ################result: {'error': 'invalid_json', 'raw_output': '```json\n{\n  "Explanation": "The Predicted Answer correctly identifies James Madison as the President of the United States in 1812, which matches the Ground Truth Answer. The substance and meaning of the Ground Truth Answer are present in the Predicted Answer.",\n  "Decision": "TRUE"\n}\n```'}
# 1it [00:03,  3.98s/it]################text: ```json
# {
#   "Explanation": "The Predicted Answer incorrectly identifies 'Sound of Music' as Disney and states that Disney was added to the S&P 500 in 1995. The Ground Truth Answer states that the company associated with 'Sound of Music' was added to the S&P 500 in 1999. The Ground Truth Answer is not present in the Predicted Answer, as the year and the company mentioned are both incorrect.",
#   "Decision": "FALSE"
# }
# ```
# ################result: {'error': 'invalid_json', 'raw_output': '```json\n{\n  "Explanation": "The Predicted Answer incorrectly identifies \'Sound of Music\' as Disney and states that Disney was added to the S&P 500 in 1995. The Ground Truth Answer states that the company associated with \'Sound of Music\' was added to the S&P 500 in 1999. The Ground Truth Answer is not present in the Predicted Answer, as the year and the company mentioned are both incorrect.",\n  "Decision": "FALSE"\n}\n```'}
# 2it [00:07,  3.59s/it]
# Judge finished


# import openai


# def judge_answer(question, gold, prediction):

#     prompt = f"""
#         You are a QA evaluator.

#         Question:
#         {question}

#         Ground Truth:
#         {gold}

#         Model Answer:
#         {prediction}

#         Is the model answer correct?

#         Output only:
#         CORRECT
#         or
#         INCORRECT
#         """

#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[{"role":"user","content":prompt}],
#         temperature=0
#     )

#     result = response["choices"][0]["message"]["content"]

#     return "CORRECT" in result.upper()
