# 基于LoRA微调的DistilBERT影评分类模型

使用LoRA技术对DistilBERT模型进行参数高效微调，实现IMDB影评文本的二分类（正面评论/负面评论）。系统包含完整的训练流程、评估模块和基于Chainlit的可视化交互界面，适用于影评情感分析场景。

## 项目结构

```
distilbert_classification/
├── README.md                              # 项目说明文档
├── chainlit_distilbert_classification.py  # 交互式分类界面(Web服务端)
├── distilbert_classification_model.py     # 模型训练与评估模块
├── training_metrics.pdf                   # 训练过程可视化图表
└── lora_model/                            # 微调后的模型组件
    ├── adapter_config.json                # LoRA适配器结构配置
    ├── adapter_model.safetensors          # 模型权重(safetensors格式)
    ├── special_tokens_map.json            # 特殊token映射表
    ├── tokenizer_config.json              # 分词器配置参数
    └── ...                                # 其他模型配置文件
└── dataset/                               # 数据管理模块
    ├── train.csv                          # 训练集
    ├── validation.csv                     # 验证集
    └── test.csv                           # 测试集
```

## 项目运行流程

- 训练微调阶段：`python distilbert_classification_model.py`
- 部署运行阶段：`chainlit run chainlit_distilbert_classification.py`
- 备注：可以直接clone项目后一键部署运行，项目内已有LoRA微调后的模型参数

## 模型训练与微调功能详解

### 1. 数据集管理

- 使用IMDB影评数据集进行训练和评估
- 数据集包含影评文本及其对应的情感标签（正面/负面）
- `IMDBDataset`类用于加载和处理数据集

```python
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
```

### 2. 分词器与模型导入

**分词器初始化与配置**
| 参数/操作                  | 作用                         | 必要性 | 注意事项                              |
|---------------------------|------------------------------|--------|---------------------------------------|
| `from_pretrained("distilbert-base-uncased")` | 加载DistilBERT预训练分词器 | ✅必需  | 需与后续模型架构严格匹配              |
| `pad_token`设置           | 定义填充标记                  | ✅关键  | 原始DistilBERT无专用pad_token，必须显式设置 |

- `distilbert_classification_model.py`分词器相关代码
```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # 初始化 DistilBERT 分词器
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad token
```

**分类模型加载**
| 参数           | 值              | 作用                              |
|----------------|-----------------|-----------------------------------|
| `"distilbert-base-uncased"` | 模型标识         | 加载66M参数的DistilBERT基础架构   |
| `num_labels`   | 2               | 添加二分类输出层(正面/负面)        |
| `pad_token_id` | `eos_token_id`  | 使模型忽略填充位置的计算           |

- `distilbert_classification_model.py`模型加载相关代码
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, pad_token_id=tokenizer.eos_token_id
)
```

### 3. LoRA微调架构

- 仅微调少量模型参数，保持大部分参数冻结
- 节省显存消耗（相比全参数微调）
- `distilbert_classification_model.py`相关LoRA代码
```python
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,           # 序列分类任务
    r=8,                                  # 低秩矩阵维度
    lora_alpha=32,                        # 缩放系数
    target_modules=["q_lin", "v_lin"],    # 修改DistilBERT的注意力机制
    lora_dropout=0.1,                     # Dropout防止过拟合
    bias="none",                          # 不训练偏置项
    modules_to_save=["pre_classifier", "classifier"]  # 解冻分类层
)
```

### 4. 训练参数配置详解
| 参数名称                      | 默认值     | 作用域        | 详细说明                                                                 |
|-------------------------------|------------|--------------|--------------------------------------------------------------------------|
| **output_dir**               | './lora_imdb' | 全局配置      | 训练输出目录（包含模型检查点、日志等）。LoRA训练时实际只保存适配器参数。   |
| **num_train_epochs**         | 3           | 训练控制      | 总训练轮次，建议设为3-5之间，根据early stopping动态调整。               |
| **per_device_train_batch_size** | 8        | 硬件资源      | 单个GPU的批次大小，建议根据显卡性能调整。                               |
| **learning_rate**            | 1e-4        | 优化器        | 初始学习率，LoRA训练建议1e-5到1e-4。                                   |
| **eval_strategy**            | "steps"     | 评估策略      | 可选"steps"（按步数）或"epoch"（每轮评估）。                           |
| **save_strategy**            | "steps"     | 模型保存      | 模型保存策略，与评估策略解耦可独立配置。                                 |
| **logging_dir**              | './logs'    | 日志系统      | TensorBoard日志存储路径，需单独启动监控服务。                            |
| **load_best_model_at_end**   | True        | 模型选择      | 训练结束后自动加载验证集最优的模型版本。                                 |
| **metric_for_best_model**    | "accuracy"  | 模型选择      | 最优模型判定指标，可改为"f1"、"precision"等。                           |

### 5. 训练、保存模型与评估预测

- `distilbert_classification_model.py`相关代码：
```python
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
```

### 6. 训练指标可视化
- `distilbert_classification_model.py`相关代码：
```python
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
```

**输入输出**
- 输入：`trainer.state.log_history`（训练日志字典列表）
- 输出：生成PDF格式的双曲线对比图：
  - 训练集 vs 验证集准确率曲线
  - 训练集 vs 验证集损失值曲线

**图表要素说明**

- 准确率对比曲线

| 元素        | 说明                          |
|-------------|-------------------------------|
| **X轴**     | 训练步数（Training Steps）     |
| **Y轴**     | 分类准确率（0-1范围）          |
| **训练曲线** | 蓝色实线（Train Accuracy）     |
| **验证曲线** | 橙色虚线（Val Accuracy）       |
| **核心功能** | 监控模型收敛/过拟合趋势        |

- 损失值对比曲线

| 元素        | 说明                          |
|-------------|-------------------------------|
| **X轴**     | 训练步数（Training Steps）     |
| **Y轴**     | 交叉熵损失值（越小越好）       |
| **训练曲线** | 蓝色实线（Train Loss）        |
| **验证曲线** | 橙色虚线（Val Loss）          |
| **核心功能** | 观察优化过程和学习率效果       |

## chainlit可视化交互界面

### 1. 模型加载模块
- 检查本地是否存在lora_model目录
- 加载基础DistilBERT分词器并配置pad_token
- 初始化基础DistilBERT分类模型(二分类)
- 加载LoRA适配器并合并到基础模型
- 将模型设置为评估模式并返回组件

```python
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
```

### 2. 文本分类模块
- 接收用户输入文本
- 使用分词器进行编码和填充(max_length=120)
- 将输入转换为PyTorch张量并送入device(CPU/GPU)
- 执行模型推理(禁用梯度计算)
- 解析logits输出得到预测类别
- 返回"正面评论"或"负面评论"分类结果

```python
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
```

### 3. Chainlit交互模块
- 异步监听用户消息输入
- 验证输入非空
- 调用分类函数获取结果
- 通过Chainlit返回格式化结果
- 捕获并显示异常信息

```python
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
```
