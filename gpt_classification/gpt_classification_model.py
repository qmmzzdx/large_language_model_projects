import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType


def create_balanced_dataset(df):
    """
    创建一个平衡的数据集(spam 和 ham 数量相等)
    """
    num_spam = df[df["Label"] == "spam"].shape[0]  # 统计 spam 数量
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123)                # 随机采样 ham
    balanced_df = pd.concat(
        [ham_subset, df[df["Label"] == "spam"]])   # 合并成平衡数据集
    return balanced_df


def random_split(df, train_frac, validation_frac):
    """
    将数据随机划分为训练集、验证集和测试集
    """
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # 打乱顺序
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


class AccuracyCallback(TrainerCallback):
    """
    自定义回调类, 在评估时计算训练集准确率并记录到日志中
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        train_pred = self.trainer.predict(
            self.trainer.train_dataset)                               # 获取训练集预测结果
        train_acc = accuracy_score(
            train_pred.label_ids, train_pred.predictions.argmax(-1))  # 计算准确率
        self.trainer.log({"train_accuracy": train_acc})               # 记录到日志
        return control                                                # 返回原有控制信号


class SpamDataset(Dataset):
    """
    自定义 SMS 垃圾短信分类数据集
    """

    def __init__(self, csv_file, tokenizer, max_length=None):
        self.data = pd.read_csv(csv_file)
        self.data["Label"] = self.data["Label"].map(
            {"ham": 0, "spam": 1})  # 标签映射
        self.tokenizer = tokenizer  # 分词器

        # 对文本进行编码
        self.encoded_texts = [
            self.tokenizer.encode(text, truncation=True,
                                  padding='max_length', max_length=max_length)
            for text in self.data["Text"]
        ]

        # 如果未指定 max_length, 则自动确定最大长度
        self.max_length = max_length or max(
            len(encoded) for encoded in self.encoded_texts)

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        # 创建注意力掩码
        attention_mask = [1] * len(encoded) + [0] * \
            (self.max_length - len(encoded))

        # 返回字典格式样本
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


def moving_average(data, window_size):
    """计算移动平均"""
    return data.rolling(window=window_size).mean()


def plot_metrics(log_history):
    """
    绘制训练过程中的准确率和损失变化趋势图
    """
    plt.figure(figsize=(12, 6))

    # 提取训练集和验证集的日志
    train_logs = [log for log in log_history if "train_accuracy" in log]
    eval_logs = [log for log in log_history if "eval_accuracy" in log]

    # 创建时间轴
    steps = [log["step"] for log in train_logs + eval_logs]

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # 绘制准确率
    ax1.plot(
        [log["step"] for log in train_logs],
        [log["train_accuracy"] for log in train_logs],
        label="Train Accuracy",
        color='blue'
    )
    ax1.plot(
        [log["step"] for log in eval_logs],
        [log["eval_accuracy"] for log in eval_logs],
        label="Val Accuracy",
        color='orange',
        linestyle='--'
    )
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # 绘制损失
    ax2.plot(
        [log["step"] for log in log_history if "loss" in log],
        [log["loss"] for log in log_history if "loss" in log],
        label="Train Loss",
        color='blue'
    )
    ax2.plot(
        [log["step"] for log in log_history if "eval_loss" in log],
        [log["eval_loss"] for log in log_history if "eval_loss" in log],
        label="Val Loss",
        color='orange',
        linestyle='--'
    )
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.pdf")  # 保存图表为 PDF 文件
    plt.close()


if __name__ == "__main__":
    # 加载原始 TSV 文件
    data_file_path = "./dataset/SMSSpamCollection.tsv"
    df = pd.read_csv(data_file_path, sep="\t",
                     header=None, names=["Label", "Text"])

    # 平衡数据集
    balanced_df = create_balanced_dataset(df)

    # 划分数据集
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # 保存分割后的数据集
    train_df.to_csv("./dataset/train.csv", index=None)
    validation_df.to_csv("./dataset/validation.csv", index=None)
    test_df.to_csv("./dataset/test.csv", index=None)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 初始化 GPT-2 分词器
    tokenizer.pad_token = tokenizer.eos_token          # 设置 pad token

    # 实例化数据集
    train_dataset = SpamDataset(
        "./dataset/train.csv", tokenizer, max_length=None)
    val_dataset = SpamDataset(
        "./dataset/validation.csv", tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(
        "./dataset/test.csv", tokenizer, max_length=train_dataset.max_length)

    # 加载 GPT-2 模型用于分类任务
    model = AutoModelForSequenceClassification.from_pretrained(
        "gpt2", num_labels=2, pad_token_id=tokenizer.eos_token_id
    )

    # 配置LoRA参数
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                                  # LoRA秩
        lora_alpha=32,                        # 缩放因子
        target_modules=["c_attn", "c_proj"],  # 针对GPT-2的注意力机制
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["score"]             # 分类层保持全参数训练
    )

    # 应用LoRA适配器
    model = get_peft_model(model, lora_config)

    # 打印可训练参数(验证配置)
    model.print_trainable_parameters()

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        report_to="none",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-4,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        label_names=["Label"],
        compute_metrics=lambda p: {"accuracy": accuracy_score(
            p.label_ids, p.predictions.argmax(-1))},
    )

    # 添加回调
    trainer.add_callback(AccuracyCallback(trainer))

    # 开始训练
    trainer.train()

    # 保存最佳模型(适配器+分类头)
    trainer.model.save_pretrained(
        "./lora_model",
        safe_serialization=True  # 使用safetensors格式
    )
    # 保存tokenizer和config
    tokenizer.save_pretrained("./lora_model")
    trainer.model.config.save_pretrained("./lora_model")

    # 验证集评估
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")

    # 测试集预测
    test_results = trainer.predict(test_dataset)
    print(f"Test results: {test_results.metrics}")

    # 生成训练指标图表
    plot_metrics(trainer.state.log_history)
