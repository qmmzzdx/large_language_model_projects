import json
import psutil
from tqdm import tqdm
import urllib.request


def query_model(prompt, model, url="http://localhost:11434/api/chat"):
    """
    向指定的模型发送请求并获取响应。

    :param prompt: 用户输入的提示
    :param model: 要使用的模型名称
    :param url: API 的 URL，默认为 http://localhost:11434/api/chat
    :return: 模型的响应内容
    """
    # 创建数据负载字典
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {  # 设置选项以确保响应的确定性
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # 将字典转换为 JSON 格式字符串并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建请求对象，设置方法为 POST 并添加必要的头部
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 发送请求并捕获响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 逐行读取和解码响应
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data


def check_if_running(process_name):
    """
    检查指定名称的进程是否正在运行。

    :param process_name: 要检查的进程名称
    :return: 布尔值，指示进程是否正在运行
    """
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def format_input(entry):
    """
    将输入数据格式化为模型所需的格式。

    :param entry: 包含指令和输入的字典
    :return: 格式化后的字符串
    """
    instruction_text = (f"<|user|>\n{entry['instruction']}")
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def generate_model_scores(json_data, json_key, model):
    """
    对模型的响应进行评分。

    :param json_data: 包含测试数据的 JSON 对象
    :param json_key: 要评分的响应键
    :param model: 要使用的模型名称
    :return: 评分列表
    """
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        # 如果模型响应为空，则评分为 0
        if entry[json_key] == "":
            scores.append(0)
        else:
            # 构建评分提示
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}` "
                f"on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            # 查询模型并获取评分
            score = query_model(prompt, model)
            try:
                scores.append(int(score))  # 尝试将评分转换为整数
            except ValueError:
                print(f"Could not convert score: {score}")  # 输出错误信息
                continue
    return scores


if __name__ == "__main__":
    # 检查 Ollama 是否正在运行
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))

    # 加载测试数据
    with open("instruction_datas/instruction-data-with-response-lora.json", "r") as file:
        test_data = json.load(file)

    model = "phi3:mini"  # 指定要使用的模型
    scores = generate_model_scores(test_data, "model_response", model)  # 生成评分
    print(f"Number of scores: {len(scores)} of {len(test_data)}")  # 输出评分数量
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")  # 输出平均分
