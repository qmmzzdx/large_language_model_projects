# 基于LoRA微调的GPT-2垃圾短信分类模型

使用LoRA技术对GPT-2模型进行参数高效微调，实现短信文本的二分类（垃圾信息/正常信息）。系统包含完整的训练流程、评估模块和基于chainlit的可视化交互界面，适用于反垃圾信息过滤场景。

## 项目结构

```
gpt_classification/
├── README.md                       # 项目说明文档
├── chainlit_gpt_classification.py  # 交互式分类界面(Web服务端)
├── gpt_classification_model.py     # 模型训练与评估模块
├── training_metrics.pdf            # 训练过程可视化图表
└── lora_model/                     # 微调后的模型组件
    ├── adapter_config.json         # LoRA适配器结构配置
    ├── adapter_model.safetensors   # 模型权重(safetensors格式)
    ├── special_tokens_map.json     # 特殊token映射表
    ├── tokenizer_config.json       # 分词器配置参数
    └── ...                         # 其他模型配置文件
└── dataset/                        # 数据管理模块
    ├── SMSSpamCollection.tsv       # 原始数据集
    ├── train.csv                   # 训练集(70%)
    ├── validation.csv              # 验证集(10%)
    └── test.csv                    # 测试集(20%)
```

## 项目运行流程

- 训练微调阶段：python gpt_classification_model.py
- 部署运行阶段：chainlit run chainlit_gpt_classification.py
- 备注：可以直接clone项目后一键部署运行，项目内已有lora微调后的模型参数

## 模型训练与微调功能详解

### 1. 数据预处理

- 解决类别不平衡问题（原始数据ham占比86.6%）
- 保持spam/ham样本比例1:1
- 固定随机种子确保可复现性
- `gpt_classification_model.py`相关代码
```python
def create_balanced_dataset(df):
    # 统计spam样本数（少数类）
    num_spam = df[df["Label"] == "spam"].shape[0]  
    # 对ham样本进行下采样
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # 合并平衡数据集
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])
```
  
### 2. 动态数据划分

- 默认比例：训练集70%，验证集10%，测试集20%
- 基于pandas的高效数据管理
- 输出标准化CSV格式便于后续处理
- `gpt_classification_model.py`相关代码
```python
def random_split(df, train_frac=0.7, validation_frac=0.1):
    df = df.sample(frac=1, random_state=123)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]
```

### 3. 分词器与模型导入

**分词器初始化与配置**
| 参数/操作                  | 作用                         | 必要性 | 注意事项                              |
|---------------------------|------------------------------|--------|---------------------------------------|
| `from_pretrained("gpt2")` | 加载GPT-2预训练分词器         | ✅必需  | 需与后续模型架构严格匹配              |
| `pad_token`设置           | 定义填充标记                  | ✅关键  | 原始GPT-2无专用pad_token，必须显式设置 |
| 使用`eos_token`           | 将序列结束符兼作填充符        | ✅推荐方案 | 需确保tokenizer.eos_token_id与模型配置一致 |
- `gpt_classification_model.py`分词器相关代码
```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 初始化 GPT-2 分词器
tokenizer.pad_token = tokenizer.eos_token          # 设置 pad token
```
**底层机制**

**1. 词汇表兼容性**
- 采用BPE(Byte Pair Encoding)分词方式
- 词汇表大小：50,257个token
- 必须与模型embedding层完全对齐

**2. 填充处理优化**
- 输入ID结构：`[token_ids] + [EOS] + [PAD,...]`
- 注意力机制会自动忽略PAD位置的计算

**分类模型加载**
| 参数           | 值              | 作用                              |
|----------------|-----------------|-----------------------------------|
| `"gpt2"`       | 模型标识         | 加载125M参数的GPT-2基础架构       |
| `num_labels`   | 2               | 添加二分类输出层(spam/ham)        |
| `pad_token_id` | `eos_token_id`  | 使模型忽略填充位置的计算           |
- `gpt_classification_model.py`模型加载相关代码
```python
# 加载 GPT-2 模型用于分类任务
model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2", num_labels=2, pad_token_id=tokenizer.eos_token_id
)
```
**架构特性**

**1. 基础架构**
- 基于GPT-2 Transformer解码器
- 12个隐藏层，768隐藏维度
- 12个注意力头

**2. 分类适配**
- 在基础模型顶部添加线性分类层
- 输出维度固定为2（二分类任务）

**3. 参数初始化**
- 主干网络保持预训练权重不变
- 分类层使用随机初始化(Lora)

### 4. LoRA微调架构

- 仅微调0.1%的模型参数（约1.1M可训练参数）
- 保持GPT-2的95%参数冻结
- 节省75%显存消耗（相比全参数微调）
- `gpt_classification_model.py`相关Lora代码
```python
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,           # 序列分类任务
    r=8,                                  # 低秩矩阵维度
    lora_alpha=32,                        # 缩放系数（控制适配器影响强度）
    target_modules=["c_attn", "c_proj"],  # 修改GPT-2的注意力机制
    lora_dropout=0.1,                     # Dropout防止过拟合
    bias="none",                          # 不训练偏置项
    modules_to_save=["score"]             # 全参数训练分类层
)
```

### 5. 训练参数配置详解
| 参数名称                      | 默认值     | 作用域        | 详细说明                                                                 |
|-------------------------------|------------|--------------|--------------------------------------------------------------------------|
| **output_dir**               | './results' | 全局配置      | 训练输出目录（包含模型检查点、日志等）。LoRA训练时实际只保存适配器参数。   |
| **report_to**                | "none"      | 日志系统      | 禁用第三方报告工具（如Weights & Biases），避免依赖外部服务。             |
| **num_train_epochs**         | 5           | 训练控制      | 总训练轮次，建议设为3-10之间，根据early stopping动态调整。               |
| **per_device_train_batch_size** | 16        | 硬件资源      | 单个GPU的批次大小，T4显卡建议8-32，A100建议32-128。                      |
| **per_device_eval_batch_size** | 16        | 硬件资源      | 验证/测试时的批次大小，通常与训练批次一致。                              |
| **learning_rate**            | 1e-4        | 优化器        | 初始学习率，LoRA训练建议1e-5到1e-4，全参数微调建议更小(5e-6)。           |
| **eval_strategy**            | "steps"     | 评估策略      | 可选"steps"（按步数）或"epoch"（每轮评估），动态评估更节省时间。         |
| **eval_steps**               | 100         | 评估策略      | 配合eval_strategy="steps"，每训练100步执行一次验证集评估。               |
| **save_strategy**            | "steps"     | 模型保存      | 模型保存策略，与评估策略解耦可独立配置。                                 |
| **save_steps**               | 100         | 模型保存      | 每100步保存一个检查点，注意磁盘空间占用。                                |
| **logging_dir**              | './logs'    | 日志系统      | TensorBoard日志存储路径，需单独启动监控服务。                            |
| **logging_strategy**         | "steps"     | 日志系统      | 日志记录频率，建议与评估步调一致便于分析。                               |
| **logging_steps**            | 100         | 日志系统      | 每100步记录一次训练指标（loss/lr等）。                                  |
| **load_best_model_at_end**   | True        | 模型选择      | 训练结束后自动加载验证集最优的模型版本。                                 |
| **metric_for_best_model**    | "accuracy"  | 模型选择      | 最优模型判定指标，可改为"f1"、"precision"等。                           |
| **greater_is_better**        | True        | 模型选择      | 指标方向，True表示指标值越大越好（如accuracy）。                        |
| **max_grad_norm**            | 1.0         | 优化器        | 梯度裁剪阈值，防止梯度爆炸，LoRA训练可适当增大（如2.0）。                |
| **lr_scheduler_type**        | "cosine"    | 学习率调度    | 可选"linear"、"cosine"、"constant"等，余弦退火cosine通常效果最佳。      |
| **warmup_ratio**             | 0.1         | 学习率调度    | 10%的训练步用于学习率线性预热，避免初期不稳定。                          |
- 每100步自动记录训练/验证准确率
- 实时追踪损失曲线
- 自动保存最优模型检查点
- `gpt_classification_model.py`训练参数相关代码
```python
training_args = TrainingArguments(
    output_dir='./results',           # 训练输出目录(lora微调时不需要)
    report_to="none",                 # 禁用第三方报告(如wandb)
    num_train_epochs=5,               # 训练轮次
    per_device_train_batch_size=16,   # 单设备训练批次大小
    per_device_eval_batch_size=16,    # 单设备评估批次大小
    learning_rate=1e-4,               # 初始学习率
    eval_strategy="steps",            # 按步数评估
    eval_steps=100,                   # 每100步评估一次
    save_strategy="steps",            # 按步数保存模型
    save_steps=100,                   # 每100步保存检查点
    logging_dir='./logs',             # 日志存储目录
    logging_strategy="steps",         # 按步数记录日志
    logging_steps=100,                # 每100步记录日志
    load_best_model_at_end=True,      # 训练结束时加载最优模型
    metric_for_best_model="accuracy", # 最优模型判定指标
    greater_is_better=True,           # 准确率越高越好
    max_grad_norm=1.0,                # 梯度裁剪阈值
    lr_scheduler_type="cosine",       # 余弦学习率调度
    warmup_ratio=0.1,                 # 10%训练步用于学习率预热
)
```

### 6. 训练器参数配置
| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|------|------|
| **model** | `PreTrainedModel` | ✅ | - | 需要训练的模型实例（LoRA 微调时为 `PeftModel` 封装后的模型） |
| **args** | `TrainingArguments` | ✅ | - | 训练参数配置对象，控制 epoch、batch size 等超参数 |
| **train_dataset** | `Dataset` | ✅ | - | 训练数据集，需实现 `__len__` 和 `__getitem__` 方法 |
| **eval_dataset** | `Dataset` | ❌ | `None` | 验证数据集（结构与训练集一致），为空时禁用评估 |
| **compute_metrics** | `Callable` | ❌ | `None` | 自定义评估函数，需返回指标字典 |
| **label_names** | `List[str]` | ❌ | `None` | 数据集中标签字段的名称（对应 CSV 中的列名） |
| **callbacks** | `List[TrainerCallback]` | ❌ | `None` | 自定义回调列表（如早停、日志增强等） |
| **optimizers** | `Tuple[Optimizer, LambdaLR]` | ❌ | `(None, None)` | 自定义优化器和学习率调度器 |
| **tokenizer** | `PreTrainedTokenizer` | ❌ | `None` | 分词器实例（用于日志记录输入样本） |
| **data_collator** | `DataCollator` | ❌ | `default_data_collator` | 批数据整理函数 |
- `gpt_classification_model.py`训练器相关代码
```python
trainer = Trainer(
    model=model,                      # 已加载LoRA的模型
    args=training_args,               # 配置好的训练参数
    train_dataset=train_dataset,      # 平衡后的训练集
    eval_dataset=val_dataset,         # 验证集
    label_names=["Label"],            # 标签字段名
    compute_metrics=lambda p: {       # 自定义评估指标
        "accuracy": accuracy_score(
            p.label_ids,              # 真实标签
            p.predictions.argmax(-1)  # 预测类别
        )
    },
)
```

### 7. 训练回调机制

**训练回调机制详解**
- 实时监控模型在训练集上的准确率变化，用于检测过拟合和训练稳定性。
- 不影响正常训练流程，仅读取模型当前状态。
- 日志数据自动兼容TensorBoard等监控工具。
- 支持多个回调的链式调用。

**回调注册机制执行流程**

**1. 初始化绑定**
- 回调实例会持有训练器(`trainer`)的引用  
- 建立双向通信通道

**2. 事件注册**
- 将回调加入训练器的事件监听队列  
- 支持多个回调的链式调用

**3. 自动触发**
| 训练阶段       | 对应回调方法     |
|---------------|------------------|
| 评估开始       | `on_evaluate()`  |
| 检查点保存     | `on_save()`      |
| 训练中断       | `on_interrupt()` |

**回调注册机制应用场景**
- 模型性能监控（如准确率/损失曲线）
- 训练过程控制（早停/学习率调整）  
- 自定义日志记录  
- 资源监控（显存/GPU利用率）
- `gpt_classification_model.py`回调AccuracyCallback相关代码：
```python
class AccuracyCallback(TrainerCallback):
      def on_evaluate(self, args, state, control, kwargs):
          # 实时计算训练集准确率
          train_pred = self.trainer.predict(self.trainer.train_dataset)
          train_acc = accuracy_score(train_pred.label_ids, 
                                    train_pred.predictions.argmax(-1))
          self.trainer.log({"train_accuracy": train_acc})
```
- 核心方法：
  - on_evaluate()：在每次评估阶段自动触发  
  - 内部使用训练器的predict()方法获取当前模型预测结果  
  - 通过accuracy_score计算预测值与真实标签的匹配度  
  - 结果通过log()方法记录到训练日志

### 8. 训练、保存模型与评估预测

- `gpt_classification_model.py`相关代码：
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
```

**训练过程与相关技术**
| 阶段         | 图标 | 说明                     | 技术细节                                                                 |
|--------------|------|--------------------------|--------------------------------------------------------------------------|
| **数据加载** | 📊   | 动态批处理与填充         | - 自动将数据分批次<br>- 应用padding统一长度<br>- 生成attention_mask       |
| **前向传播** | ⏩   | 计算模型输出             | - 输入通过所有网络层<br>- 输出分类logits<br>- 使用当前参数计算            |
| **损失计算** | 📉   | 交叉熵损失函数           | - 比较预测与真实标签<br>- 计算分类误差<br>- 公式: L = -Σy*log(p)          |
| **反向传播** | 🔙   | 自动微分求梯度           | - 计算各参数梯度<br>- 链式法则逐层传播<br>- 记录计算图                    |
| **参数更新** | 🔄   | Adam优化器调整参数       | - 自适应学习率调整<br>- 动量加速收敛                                     |
| **评估检查点** | ✅   | 验证集性能监控           | - 计算验证集指标<br>- 保存最佳模型<br>- 早停机制检测                    |

**保存模型**
| 文件名 | 文件类型 | 内容描述 | 是否必需 | 安全特性 |
|--------|----------|----------|----------|----------|
| **adapter_config.json** | 配置文件 | 包含LoRA配置参数：<br>- 秩(r)<br>- 缩放因子(alpha)<br>- 目标模块列表 | ✅ | JSON Schema验证 |
| **adapter_model.safetensors** | 权重文件 | 二进制格式存储：<br>- LoRA适配器权重<br>- 分类头参数 | ✅ | Safetensors加密 |
| **config.json** | 配置文件 | 基础模型架构：<br>- 隐藏层维度<br>- 注意力头数<br>- pad_token_id | ✅ | 版本签名校验 |
| **tokenizer_config.json** | 配置文件 | 分词处理规则：<br>- padding策略<br>- truncation设置<br>- 特殊token定义 | ✅ | 编码规范检查 |
| **tokenizer.json** | 分词器文件 | 新格式整合：<br>- 词汇表<br>- 合并规则<br>- 分词算法配置 | ✅ | 完整性哈希 |
| **special_tokens_map.json** | 映射文件 | 特殊token对应关系：<br>- [PAD]<br>- [CLS]<br>- [SEP] | ❌ | UTF-8强制 |
| **vocab.json** | 词表文件 | 旧格式词表：<br>- token到ID的映射 | ❌（兼容旧版需要） | 键值校验 |
| **merges.txt** | 规则文件 | BPE合并规则：<br>- 字符级合并顺序 | ❌（BPE分词器需要） | 行格式校验 |

**验证与测试**
- 验证集评估指标
  
| 指标名称                  | 类型    | 说明                                                                 |
|---------------------------|---------|----------------------------------------------------------------------|
| `eval_loss`               | float   | 验证集上的平均交叉熵损失值                                           |
| `eval_accuracy`           | float   | 分类准确率（当配置compute_metrics时）                               |
| `eval_runtime`            | float   | 评估总耗时（秒）                                                    |
| `eval_samples_per_second` | float   | 每秒处理的样本数（吞吐量）                                           |
| `eval_steps_per_second`   | float   | 每秒完成的批次数                                                    |

- 测试集预测输出

| 属性/指标              | 类型       | 说明                                                                 |
|------------------------|------------|----------------------------------------------------------------------|
| `predictions`          | np.ndarray | 原始预测logits（形状：[样本数×类别数]）                              |
| `label_ids`            | np.ndarray | 真实标签（形状：[样本数]）                                           |
| `metrics`              | dict       | 包含以下测试指标：                                                  |
| ↳ `test_loss`          | float      | 测试集上的平均损失值                                                |
| ↳ `test_accuracy`      | float      | 分类准确率（当配置compute_metrics时）                               |
| ↳ `test_samples`       | int        | 测试样本总数 

### 9. 训练指标可视化
- `gpt_classification_model.py`相关代码：
```python
# 生成训练指标图表
plot_metrics(trainer.state.log_history)
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
- 加载基础GPT-2分词器并配置pad_token
- 初始化基础GPT-2分类模型(二分类)
- 加载LoRA适配器并合并到基础模型
- 将模型设置为评估模式并返回组件
- `chainlit_gpt_classification.py`相关代码：
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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为pad_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "gpt2",
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
- 返回"垃圾信息"或"正常信息"分类结果
- `chainlit_gpt_classification.py`相关代码：
```python
def classify_review(user_input):
    """
    对用户输入的文本进行垃圾短信分类(spam/ham)
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

    return "垃圾信息" if predicted_class == 1 else "正常信息"
```

### 3. Chainlit交互模块
- 异步监听用户消息输入
- 验证输入非空
- 调用分类函数获取结果
- 通过Chainlit返回格式化结果
- 捕获并显示异常信息
- `chainlit_gpt_classification.py`相关代码：
```python
@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit 的主消息处理函数
    接收用户输入, 并返回分类结果
    """
    user_input = message.content

    if not user_input:
        await chainlit.Message(content="请输入一段文本以进行分类").send()
        return

    try:
        label = classify_review(user_input)
        await chainlit.Message(content=f"该文本被分类为: {label}").send()
    except Exception as e:
        await chainlit.Message(content=f"错误信息: {str(e)}").send()
```
