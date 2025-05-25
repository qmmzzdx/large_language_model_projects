import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType


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


# IMDB分类数据集class
class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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
    # 加载模型和分词器
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                                              # LoRA秩
        lora_alpha=32,                                    # 缩放因子
        target_modules=["q_lin", "v_lin"],                # DistilBERT的注意力层
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["pre_classifier", "classifier"]  # 解冻分类层
    )
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    model.print_trainable_parameters()

    # 加载本地数据集(训练集、验证集和测试集)
    train_dataset = IMDBDataset(
        csv_file='./dataset/train.csv', tokenizer=tokenizer)
    val_dataset = IMDBDataset(
        csv_file='./dataset/validation.csv', tokenizer=tokenizer)
    test_dataset = IMDBDataset(
        csv_file='./dataset/test.csv', tokenizer=tokenizer)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./lora_imdb",
        report_to="none",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=3000,
        save_strategy="steps",
        save_steps=3000,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=3000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

    # 创建 Trainer 对象并训练模型
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
