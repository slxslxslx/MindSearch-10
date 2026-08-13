# 在文件最顶部添加（第1-3行）
import re

from dotenv import load_dotenv

load_dotenv()  # 自动加载 .env 文件

import datetime  # 如果需要datetime，单独导入，不要覆盖time
import json
import os
import time  # 导入系统time模块（必须保留）
from copy import deepcopy
from pathlib import Path

from lagent.utils import create_object
from tqdm import tqdm

from ..agent import models as llm_factory
from ..agent.mindsearch_agent import MindSearchAgent

# 基于脚本位置定义路径（确保无论在哪里运行都能找到文件）
CURRENT_FILE = Path(__file__).resolve()
MINDSEARCH_DIR = CURRENT_FILE.parent.parent  # mindsearch 目录
DATA_PATH = str(MINDSEARCH_DIR / "data") + "/"
RESULT_PATH = str(MINDSEARCH_DIR / "results") + "/"

print(f"Data path: {DATA_PATH}")  # 调试时可以确认路径是否正确


# DATA_PATH = "../data/"
# RESULT_PATH = "../results/"


# def load_dataset(path):
#     with open(path, "r", encoding="utf8") as f:
#         return json.load(f)


def load_dataset(path):
    """
    读取 JSONL (JSON Lines) 格式的文件
    每行是一个独立的 JSON 对象，最终返回一个包含所有对象的列表
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        # 逐行读取并解析
        for line_num, line in enumerate(f, 1):
            # 去除行首尾的空白字符（换行、空格等）
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            try:
                # 解析单行 JSON
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                # 捕获单行解析错误，提示具体行号，方便排查
                print(f"警告：第 {line_num} 行 JSON 解析失败: {e}")
                continue
    return data


def save_results(results, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# 使用 Python 的 argparse模块 解析命令行参数
def parse_arguments():  
    import argparse  # 懒加载

    parser = argparse.ArgumentParser(description="MindSearch API")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Service host")  # 监听所有网络接口
    parser.add_argument("--port", default=8002, type=int, help="Service port")
    parser.add_argument("--lang", default="en", type=str, help="Language")
    parser.add_argument("--model_format", default="internlm_server", type=str, help="Model format")
    parser.add_argument("--search_engine", default="GoogleSearch", type=str, help="Search engine")
    parser.add_argument("--asy", default=False, action="store_true", help="Agent mode") # action="store_true",将 --asy 参数设计为开关型布尔标志（flag），而非需要赋值的参数。
    parser.add_argument("--out_folder", default="GoogleSearch_qwen-3.5-9b_bamboogle", type=str, help="Search engine")
    return parser.parse_args() # 解析命令行参数，并返回一个 Namespace 对象（里面包含所有参数的值）


args = parse_arguments()

import os

from lagent.actions import AsyncWebBrowser, WebBrowser
from lagent.agents.stream import get_plugin_prompt
from lagent.llms import INTERNLM2_META, LMDeployServer
from lagent.prompts import (  # 从lagent的提示词模块导入两个解析器类。解析<|action_start|><|plugin|>等特殊标记，提取出工具名和参数，然后去真正执行。# InterpreterParser：解析代码解释器（如 Python 执行）的输出格式，让 AI 的思考和结果按固定格式返回。这个agent会编写代码，执行代码，根据代码执行结果来回答。# PluginParser：解析工具调用环节的输出格式，让大模型知道 “怎么调用搜索工具、返回什么格式的搜索结果”。
    InterpreterParser,
    PluginParser,
)

from mindsearch.agent.mindsearch_prompt import (
    # EXTRACT_PROMPT,
    FINAL_RESPONSE_CN,
    FINAL_RESPONSE_EN,
    GRAPH_PROMPT_CN,
    GRAPH_PROMPT_EN,
    searcher_context_template_cn,
    searcher_context_template_en,
    searcher_input_template_cn,
    searcher_input_template_en,
    searcher_system_prompt_cn,
    searcher_system_prompt_en,
)


# def extract_short_answer(question, prediction, llm_client):
#     EXTRACT_PROMPT = """Given the following question and a long-form answer, extract the shortest possible answer span.
# Your output must be ONLY the answer itself, no explanation, no punctuation around it.

# Question: {question}
# Long-form answer: {prediction}

# Short answer:""".format(question=question, prediction=prediction)

#     # ✅ lagent GPTAPI.chat() 需要消息列表，不是字符串
#     messages = [{"role": "user", "content": EXTRACT_PROMPT}]
#     response = llm_client.chat(messages)
#     print(f"extract_short_answer里面的 response:{response}")
    
#     # response是字符串或AgentMessage，取content
#     if hasattr(response, "content"):
#         print(f"extract_short_answer里面的 response.content.strip():{response.content.strip()}")
#         return response.content.strip()
#     return str(response).strip()


def extract_short_answer(question, prediction, llm_client):
    # ✅ 英文版优化提示词：强调精确匹配、加入示例、明确输出格式
    EXTRACT_PROMPT = """You are a precise answer extractor. Your task is to extract the shortest possible exact text span from the provided long-form answer that directly answers the question.

Rules:
1. Output ONLY the extracted text span. 
2. Do NOT include any explanations, conversational filler, or prefixes (e.g., "The answer is", "Answer:").
3. Do NOT add trailing punctuation (e.g., periods, quotes) unless it is an integral part of the answer (e.g., decimals, titles).
4. If the answer is not present in the text, output exactly "None".

Examples:
Question: Who proposed the theory of relativity?
Long-form answer: Albert Einstein, one of the greatest physicists of the 20th century, proposed the special and general theories of relativity, fundamentally changing our understanding of the universe.
Extracted answer: Albert Einstein

Question: What is the speed of light in a vacuum?
Long-form answer: According to modern physics, the speed of light in a vacuum is a physical constant denoted by the letter c, which is approximately 300,000 kilometers per second, or 3×10^8 m/s.
Extracted answer: 3×10^8 m/s

Now extract the answer for the following:
Question: {question}
Long-form answer: {prediction}
Extracted answer:
""".format(question=question, prediction=prediction)

    # lagent GPTAPI.chat() 需要消息列表
    messages = [{"role": "user", "content": EXTRACT_PROMPT}]
    response = llm_client.chat(messages)
    
    # 兼容不同的返回类型 (字符串 或 AgentMessage对象)
    if hasattr(response, "content"):
        raw_answer = response.content.strip()
    else:
        raw_answer = str(response).strip()
    
    print(f"extract_short_answer Raw response: {raw_answer}")
    
    # ✅ 防御性编程：正则兜底，防止大模型偶尔不听话
    # 1. 去除常见英文前缀
    cleaned_answer = re.sub(
        r'^(Extracted answer:\s*|The answer is\s*|Answer:\s*)', 
        '', raw_answer, flags=re.IGNORECASE
    )
    # 2. 只取第一行，防止它后面又换行解释
    final_answer = cleaned_answer.split('\n')[0].strip()
    # 3. 去除首尾可能残留的标点（引号、句号等，保留小数点和连字符）
    final_answer = re.sub(r'^[\s""\"\"\'\'\[\(]+|[\s""\"\"\'\'\]\),.!?;:]+$', '', final_answer)
    print(f"extract_short_answer里面的 final_answer:{final_answer}")
    
    return final_answer


def strip_think(text):
    """去除 <think>...</think> 及其内容，返回实际答案"""
    print(f"strip_think里面的 text:{text}")
    if not text:
        return text
    # 匹配 </think> 之后的所有内容
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def run_agent_on_dataset(dataset_name):

    dataset = load_dataset(DATA_PATH + dataset_name)

    lang = args.lang  #
    date = datetime.datetime.now().strftime("The current date is %Y-%m-%d.")
    # llm = LMDeployServer(
    #     path="/data1/llm/internlm2_5-7b-chat",  # "internlm/internlm2_5-7b-chat",
    #     model_name="internlm2_5-7b-chat",  # "internlm2",
    #     meta_template=INTERNLM2_META,
    #     top_p=0.8,
    #     top_k=1,
    #     temperature= 1.0, # 0,  # 1.0,
    #     max_new_tokens=8192,  # 4096, # 8192
    #     repetition_penalty=1.02,
    #     stop_words=["<|im_end|>", "<|action_end|>"], # ["<|im_end|>"],  # stop_words=["<|im_end|>", "<|action_end|>"],
    # )

    LLM = {}
    mode = "async" if args.asy else "sync"
    llm = LLM.get(args.model_format, {}).get(mode)
    if llm is None:
        llm_cfg = deepcopy(getattr(llm_factory, args.model_format))
        if llm_cfg is None:
            raise NotImplementedError
        if args.asy:
            cls_name = (
                llm_cfg["type"].split(".")[-1] if isinstance(
                    llm_cfg["type"], str) else llm_cfg["type"].__name__)
            llm_cfg["type"] = f"lagent.llms.Async{cls_name}"
        llm = create_object(llm_cfg) # <lagent.llms.lmdeploy_wrapper.AsyncLMDeployServer object at 0x7ff612277bb0>
        LLM.setdefault(args.model_format, {}).setdefault(mode, llm) # {'internlm_server': {'async': <lagent.llms.lmdeploy_wrapper.AsyncLMDeployServer object at 0x7ff612277bb0>}}

    use_async = False
    # use_async = True
    search_engine = args.search_engine  #SerpBingSearch # "BingSearch" #"GoogleSearch" # "TencentSearch"  # "DuckDuckGoSearch"  # 确保拼写完全匹配（区分大小写！）

    # ✅ 精简清晰的插件配置（无元组、无冗余参数）
    if search_engine == "TencentSearch":
        plugin_config = {
            "type": WebBrowser,  # use_async=False 时直接写 WebBrowser
            # "type": AsyncWebBrowser,
            "searcher_type": search_engine,
            "topk": 6,
            "secret_id": os.getenv("TENCENT_SEARCH_SECRET_ID"),
            "secret_key": os.getenv("TENCENT_SEARCH_SECRET_KEY"),
        }
    elif search_engine == "DuckDuckGoSearch":
        plugin_config = {
            "type": WebBrowser,
            # "type": AsyncWebBrowser,
            "searcher_type": search_engine,
            "topk": 6,
            # ⚠️ DuckDuckGoSearch 不需要任何密钥！BingSearch 需额外加 api_key
            # "api_key": os.getenv("BING_API_KEY")  # 仅当 search_engine=="BingSearch" 时取消注释
        }
    elif search_engine in ["BingSearch", "BraveSearch", "GoogleSearch"]:
        env_var_name = {
            "BingSearch": "BING_SEARCH_API_KEY",
            "BraveSearch": "BRAVE_SEARCH_API_KEY",
            "GoogleSearch": "GOOGLE_SERPER_API_KEY",
        }[search_engine]
        plugin_config = {
            "type": WebBrowser,
            # "type": AsyncWebBrowser,
            "gl": "us",  # 国家
            "hl": "en",  # 语言
            "searcher_type": search_engine,
            "topk": 6,
            "api_key": os.getenv(env_var_name),
        }
    elif search_engine == "SerpBingSearch":
        plugin_config = {
            "type": WebBrowser,
            "searcher_type": "SerpBingSearch",
            "mkt": "en-US",
            "topk": 6,
            "api_key": os.getenv("SERPBING_API_KEY"),   # .env 里设置这个变量
        }

    plugins = [plugin_config]  # ✅ 直接字典列表，无元组嵌套

    # agent = MindSearchAgent(
    #     llm=llm,
    #     template=date,
    #     output_format=InterpreterParser(
    #         template=GRAPH_PROMPT_CN if lang == "cn" else GRAPH_PROMPT_EN
    #     ),
    #     searcher_cfg=dict(
    #         llm=llm,
    #         plugins=plugins,
    #         template=date,
    #         output_format=PluginParser(
    #             template=(
    #                 searcher_system_prompt_cn
    #                 if lang == "cn"
    #                 else searcher_system_prompt_en
    #             ),
    #             tool_info=get_plugin_prompt(plugins),
    #         ),
    #         user_input_template=(
    #             searcher_input_template_cn
    #             if lang == "cn"
    #             else searcher_input_template_en
    #         ),
    #         user_context_template=(
    #             searcher_context_template_cn
    #             if lang == "cn"
    #             else searcher_context_template_en
    #         ),
    #     ),
    #     summary_prompt=FINAL_RESPONSE_CN if lang == "cn" else FINAL_RESPONSE_EN,
    #     max_turn=10,  # 6,  # 10,
    # )

    results = []

    # 生成格式化的时间字符串（年-月-日_时-分-秒）
    # time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # # 拼接文件名：数据集名 + 时间 + 后缀
    # file_name = dataset_name.replace(
    #     ".json", f"_Tencent_12P_{time_str}.json"
    # )
    # full_path = os.path.join( RESULT_PATH, file_name )  # 推荐用os.path.join拼接路径，避免跨平台问题
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_file = RESULT_PATH / args.out_folder / f"{args.out_folder}_{time_str}.jsonl"
    out_file = os.path.join(RESULT_PATH, args.out_folder, f"{args.out_folder}_{time_str}.jsonl")

    with open(out_file, "w", encoding="utf-8") as f:

        for idx, sample in tqdm(enumerate(dataset, start=1), total=len(dataset)):

            # ✅ 每条数据重新实例化 agent，彻底避免上下文累积
            agent = MindSearchAgent(
                llm=llm,
                template=date,
                output_format=InterpreterParser(template=GRAPH_PROMPT_CN if lang == "cn" else GRAPH_PROMPT_EN),
                searcher_cfg=dict(
                    llm=llm,
                    plugins=plugins,
                    template=date,
                    output_format=PluginParser( # 
                        template=searcher_system_prompt_cn  if lang == "cn" else searcher_system_prompt_en,
                        tool_info=get_plugin_prompt(plugins), # 
                    ),
                    user_input_template=(searcher_input_template_cn if lang == "cn"
                                        else searcher_input_template_en),
                    user_context_template=(searcher_context_template_cn if lang == "cn"
                                        else searcher_context_template_en),
                ),
                summary_prompt=FINAL_RESPONSE_CN
                if lang == "cn" else FINAL_RESPONSE_EN,
                max_turn=10 # 6, #6,  # 10,
            )

            id = sample.get("id", "N/A")  # 取 id，没有就显示 N/A
            question = sample["question"]
            answer = sample.get("answer", "")

            print(f"\n______________________________当前处理第 {id} 条数据_________________________________________")

            prediction = ""

            # try:
            #     for msg in agent.forward(question):
            #         print(f"Model state: {msg.stream_state}, Response: {msg.content}")  # 实时打印模型状态和响应内容
            #         if msg.stream_state == 0:   # END
            #             prediction = msg.content

            # except Exception as e:
            #     prediction = "ERROR"

            try:
                full_response = ""  # 累积完整回答
                for msg in agent.forward(question):
                    # print(f"Model state: {msg.stream_state}")
                    # print(f"Model state: {msg.stream_state}, Response: {msg.content}")

                    # ✅ 正确逻辑：累积所有内容 + 捕获结束状态
                    if msg.stream_state in [0, 1]:  # 0=结束, 1=流式内容 # 这不对吧
                        full_response = msg.content  # 覆盖更新（因每次返回完整内容）

                    # 可选：显式检测结束状态
                    # print(f"检测结束状态，msg.stream_state={msg.stream_state}")
                    if msg.stream_state == 0:  # 实际结束标识
                        break

                prediction = strip_think(full_response.strip()) # 清理空格+ </think>
                # short_answer = strip_think(extract_short_answer(question, prediction, llm))
                print(
                    f"######################################################Prediction for question '{question}': {prediction}"
                )

            except Exception as e:
                import traceback

                print(f"⚠️ Error processing question: {e}")
                print(traceback.format_exc())  # ✅ 打印完整堆栈
                prediction = "ERROR"
                short_answer = "ERROR"     # ← 加这一行

            # ✅ 每条立即写入，用 JSON Lines 格式，无论成功还是报错，都写入，条数始终和数据集一致，后续评测不会出现索引错位的问题
            print(
                f"########################################################正在保存第 {idx} 条结果..."
            )
            record = {
                "id": id,
                "question": question,
                "answer": answer,
                "prediction": prediction,
                # "short_answer": short_answer,   # 用于EM计算
            }  # record = {"question": question, "answer": answer, "prediction": prediction}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # ← 加这一行,这样每处理完一条数据就会立即写入磁盘，不会等到整个循环结束才写。这在长时间运行的任务中尤其重要——万一中途崩溃，已处理的数据不会丢失。
            print(f"结果已保存到：{out_file}")

            # results.append({
            #     "question": question,
            #     "answer": answer,
            #     "prediction": prediction
            # })

    # save_results(results, RESULT_PATH + dataset_name.replace(".json","_predictions.json"))

    # # 生成格式化的时间字符串（年-月-日_时-分-秒）
    # time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # # 拼接文件名：数据集名 + 时间 + 后缀
    # file_name = dataset_name.replace(".json", f"_predictions_{time_str}.json")
    # full_path = os.path.join(RESULT_PATH, file_name)  # 推荐用os.path.join拼接路径，避免跨平台问题

    # # 保存结果（调用你的函数）
    # save_results(results, full_path)


if __name__ == "__main__":

    # run_agent_on_dataset("bamboogle_data.jsonl")
    run_agent_on_dataset("seal-0.jsonl")
    # run_agent_on_dataset("xbench-ds-old.jsonl")
