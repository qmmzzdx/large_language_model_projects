import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from functools import partial
from transformers import GPT2Model

from src.gpt_model import GPTModel
from src.generate_text import generate_text
from src.gpt_training import (
    load_gpt_config, text_to_token_ids, token_ids_to_text, calc_loss_loader, train_model
)


class InstructionDatasetPhi(Dataset):
    """
    自定义数据集类，用于处理指令数据集

    参数:
        data (list): 输入数据列表，每个元素是一个字典，包含指令和输出
        tokenizer: 用于将文本编码为token的分词器
    """

    def __init__(self, data, tokenizer):
        self.data = data

        # 预先对文本进行编码
        self.encoded_texts = []
        for entry in data:
            # 使用 format_input_phi 函数格式化输入
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        """
        获取指定索引的编码文本

        参数:
            index (int): 数据索引

        返回:
            list: 编码后的文本
        """
        return self.encoded_texts[index]

    def __len__(self):
        """
        获取数据集的长度

        返回:
            int: 数据集的样本数量
        """
        return len(self.data)


class LoRALayer(torch.nn.Module):
    """
    LoRA层，用于在模型中引入低秩适配

    参数:
        in_dim (int): 输入维度
        out_dim (int): 输出维度
        rank (int): 低秩矩阵的秩
        alpha (float): 缩放因子
    """

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        # 使用标准权重初始化
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        """
        前向传播方法，计算LoRA层的输出

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    """
    带有LoRA的线性层

    参数:
        linear (torch.nn.Linear): 原始线性层
        rank (int): 低秩矩阵的秩
        alpha (float): 缩放因子
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        """
        前向传播方法，计算带有LoRA的线性层的输出

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    """
    替换模型中的线性层为带有LoRA的线性层

    参数:
        model (torch.nn.Module): 需要替换的模型
        rank (int): 低秩矩阵的秩
        alpha (float): 缩放因子
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 用带有LoRA的线性层替换原始线性层
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归应用到子模块
            replace_linear_with_lora(module, rank, alpha)


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    """
    自定义数据加载函数，用于处理批量数据

    参数:
        batch (list): 输入数据的批量
        pad_token_id (int): 填充token的ID
        ignore_index (int): 忽略的索引
        allowed_max_length (int): 允许的最大序列长度
        device (str): 目标设备

    返回:
        tuple: 输入和目标的张量
    """
    # 找到批量中最长的序列
    batch_max_length = max(len(item)+1 for item in batch)

    # 填充和准备输入和目标
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 添加 endoftext 到序列末尾
        new_item += [pad_token_id]
        # 填充序列到最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # 输入为填充后的序列，去掉最后一个token，目标序列向右移动1个单位
        inputs, targets = torch.tensor(padded[:-1]), torch.tensor(padded[1:])
        # 替换目标中除第一个填充token外的所有填充token为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 可选：截断到最大序列长度
        if allowed_max_length is not None:
            inputs, targets = inputs[:allowed_max_length], targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标列表转换为张量并转移到目标设备
    inputs_tensor, targets_tensor = torch.stack(inputs_lst).to(
        device), torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def format_input_phi(entry):
    """
    格式化输入文本

    参数:
        entry (dict): 包含指令和输入的字典

    返回:
        str: 格式化后的输入文本
    """
    instruction_text = (f"<|user|>\n{entry['instruction']}")
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def load_instruction_file(file_path):
    """
    加载指令文件

    参数:
        file_path (str): 文件路径

    返回:
        list: 加载的数据
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def split_data(data, train_pos, test_pos):
    """
    将数据集分为训练集、验证集和测试集

    参数:
        data (list): 输入数据
        train_pos (int): 训练集的结束位置
        test_pos (int): 测试集的开始位置

    返回:
        tuple: 训练集、验证集和测试集
    """
    return data[:train_pos], data[train_pos + test_pos:], data[train_pos:train_pos + test_pos]


def plot_training_curves(train_losses, val_losses, tokens_seen, lrs):
    """
    绘制训练过程中的损失曲线、已见token数和学习率变化曲线

    参数:
        train_losses (list): 训练集的损失值列表
        val_losses (list): 验证集的损失值列表
        tokens_seen (list): 每个 epoch 后模型已见过的token数量
        lrs (list): 每个 epoch 的学习率列表
    """
    # 创建一个包含两个子图的图形对象, figsize 设置图形大小
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 第一个子图: 绘制训练和验证损失曲线
    ax1.plot(train_losses, label='Train', marker='o')
    ax1.plot(val_losses, label='Validation', marker='x')
    ax1.set_title('Training Progress')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    ax1.legend()

    # 第二个子图: 绘制已见令牌数和学习率变化曲线
    ax2.plot(tokens_seen, color='tab:blue', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Tokens Seen', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 双 Y 轴: 在同一子图中绘制学习率变化曲线
    ax2_lr = ax2.twinx()
    ax2_lr.plot(lrs, color='tab:red', linestyle='--', marker='^')
    ax2_lr.set_ylabel('Learning Rate', color='tab:red')
    ax2_lr.tick_params(axis='y', labelcolor='tab:red')

    plt.tight_layout()
    plt.savefig("train_metrics/finetune_training_metrics.pdf")
    plt.close()


if __name__ == "__main__":
    # 指定指令数据文件的路径
    file_path = "instruction_datas/instruction-data.json"

    # 加载指令数据
    data = load_instruction_file(file_path)

    # 将数据集分为训练集（85%）、验证集（5%）和测试集（10%）
    train_data, val_data, test_data = split_data(
        data, int(len(data) * 0.85), int(len(data) * 0.1))

    print(50*"-")
    print("Training set length:", len(train_data))  # 打印训练集长度
    print("Validation set length:", len(val_data))  # 打印验证集长度
    print("Test set length:", len(test_data))        # 打印测试集长度
    print(50*"-")

    # 获取GPT-2的分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 设置设备为GPU（如果可用）或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)  # 打印当前使用的设备
    print(50*"-")

    # 设置随机种子以确保可重复性
    torch.manual_seed(123)

    # 定义数据加载的参数
    num_workers, batch_size, allowed_max_length = 0, 8, 1024

    # 自定义数据加载函数
    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=allowed_max_length)

    # 创建训练集和验证集的数据集对象
    train_dataset = InstructionDatasetPhi(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,            # 随机打乱训练数据
        drop_last=True,          # 如果最后一个batch不满，丢弃它
        num_workers=num_workers  # 数据加载的工作线程数
    )

    val_dataset = InstructionDatasetPhi(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,  # 验证数据不需要打乱
        drop_last=False,
        num_workers=num_workers
    )

    # 定义可用的 GPT 模型及其对应的预训练模型名称
    model_names = {
        "gpt2-small (124M)": "openai-community/gpt2",          # 小型 GPT-2 模型
        "gpt2-medium (355M)": "openai-community/gpt2-medium",  # 中型 GPT-2 模型
        "gpt2-large (774M)": "openai-community/gpt2-large",    # 大型 GPT-2 模型
        "gpt2-xl (1558M)": "openai-community/gpt2-xl"          # 完整 GPT-2 模型
    }

    # 当前选择大型 GPT-2 模型
    CHOOSE_MODEL = "gpt2-large (774M)"

    # 从 JSON 文件中加载 GPT 配置
    gpt_config = load_gpt_config(model_name='GPT_CONFIG_774M')

    # 从 Hugging Face 加载预训练的 GPT 模型
    gpt_hf = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL])
    gpt_hf.eval()  # 设置模型为评估模式

    # 创建自定义模型并加载预训练权重
    model = GPTModel(gpt_config)
    model.load_weights(gpt_config, gpt_hf)
    model = model.to(device)  # 将模型转移到指定设备
    model.eval()  # 设置自定义模型为评估模式

    print("Loaded model:", CHOOSE_MODEL)  # 打印加载的模型名称
    print(50*"-")

    # 计算并打印可训练参数的总数
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters before: {total_params:,}")

    # 冻结所有参数以便只训练LoRA参数
    for param in model.parameters():
        param.requires_grad = False

    # 计算并打印冻结后的可训练参数总数
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters after: {total_params:,}")

    # 用LoRA替换线性层
    replace_linear_with_lora(model, rank=16, alpha=16)

    # 计算并打印LoRA参数的总数
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Total trainable LoRA parameters: {total_params:,}")
    model.to(device)  # 确保模型在正确的设备上
    print(50*"-")

    print("Initial losses")
    with torch.no_grad():
        # 计算训练集和验证集的初始损失
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss:", train_loss)  # 打印训练损失
    print("Validation loss:", val_loss)  # 打印验证损失
    print(50*"-")

    start_time = time.time()  # 记录训练开始时间

    # 最大学习率
    peak_lr = 6e-5
    # 总共训练 2 个 epoch
    num_epochs = 2
    # 总训练步数
    total_steps = len(train_loader) * num_epochs
    # 前 20% 的步数用于学习率预热
    warmup_steps = int(0.2 * total_steps)

    # 定义优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, weight_decay=0.1)

    # 获取验证集的第一个输入上下文
    start_context = format_input_phi(val_data[0])

    # 训练模型并获取损失和学习率变化
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device,
        n_epochs=num_epochs, eval_freq=5, eval_iter=5,
        warmup_steps=warmup_steps, start_context=start_context, tokenizer=tokenizer
    )

    end_time = time.time()  # 记录训练结束时间
    execution_time_minutes = (end_time - start_time) / 60  # 计算训练时间（分钟）
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 绘制训练曲线
    plot_training_curves(
        train_losses=train_losses,  # 训练损失列表
        val_losses=val_losses,      # 验证损失列表
        tokens_seen=tokens_seen,    # 已见token数
        lrs=lrs                     # 学习率列表
    )
    print(50*"-")

    print("Generating responses...")
    # 生成测试集的响应
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input_phi(entry)  # 格式化输入文本
        # 获取模型的最大上下文长度
        context_size = model.pos_emb.weight.shape[0]
        token_ids = generate_text(
            model=model, idx=text_to_token_ids(
                input_text, tokenizer).to(device),
            max_new_tokens=256, context_size=context_size, temperature=0.2, top_k=5, eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        # 提取模型生成的响应文本
        response_text = generated_text[len(input_text):].replace(
            "<|assistant|>:", "").strip()
        test_data[i]["model_response"] = response_text  # 将响应添加到测试数据中

    # 保存生成的响应到文件
    test_data_path = "instruction_datas/instruction-data-with-response-lora.json"
    file_name = f"lora_model/gpt-sft-lora.pth"

    # 将测试数据写入JSON文件
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")

    # 只保存LoRA相关的参数
    lora_state_dict = {k: v for k, v in model.state_dict().items()
                       if 'lora' in k.lower()}
    torch.save(lora_state_dict, file_name)
    print(f"Model saved as {file_name}")
