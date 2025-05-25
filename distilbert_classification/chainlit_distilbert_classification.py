import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import chainlit

# 设置运行设备(GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载本地保存的LoRA微调模型和分词器
    如果模型目录不存在, 则输出错误并退出程序
    """
    model_path = './lora_model'  # LoRA适配器

    if not os.path.exists(model_path):
        print(f"找不到 {model_path} 目录, 请确认模型已正确保存")
        sys.exit(1)

    # 加载基础模型和分词器
    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 确保填充token设置(针对DistilBERT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        pad_token_id=tokenizer.eos_token_id
    )

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, model_path)

    # 合并适配器到基础模型
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    return tokenizer, model


def classify_review(user_input):
    """
    对用户输入的文本进行IMDB影评分类(Positive/Negative)
    """

    tokenizer, model = get_model_and_tokenizer()

    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=120
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return "正面评论" if predicted_class == 1 else "负面评论"


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit 的主消息处理函数
    接收用户输入, 并返回分类结果
    """

    user_input = message.content

    if not user_input:
        await chainlit.Message(content="请输入一段影评以进行分类").send()
        return

    try:
        label = classify_review(user_input)
        await chainlit.Message(content=f"该文本被分类为: {label}").send()
    except Exception as e:
        await chainlit.Message(content=f"错误信息: {str(e)}").send()
