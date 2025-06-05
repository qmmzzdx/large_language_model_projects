from pathlib import Path
import json
import sys
import tiktoken
import torch
import chainlit

from src.gpt_model import GPTModel
from src.generate_text import generate_text
from src.gpt_training import load_gpt_config, text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载GPT模型和tokenizer
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    gpt_config = load_gpt_config(model_name='GPT_CONFIG_774M')

    # 加载 LoRA 权重
    lora_model_path = Path("lora_model/gpt-sft-lora.pth")
    if not lora_model_path.exists():
        print(
            f"Could not find the {lora_model_path} file. Please ensure the LoRA model is saved.")
        sys.exit(1)

    lora_checkpoint = torch.load(
        lora_model_path, weights_only=True, map_location=device)
    model = GPTModel(gpt_config)
    model.load_state_dict(lora_checkpoint, strict=False)  # 加载LoRA权重，允许不严格匹配
    model.to(device)
    return tokenizer, model, gpt_config


def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("<|assistant|>:", "").strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, gpt_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    torch.manual_seed(123)

    prompt = f"<|user|>\n{message.content}"

    token_ids = generate_text(  # function uses `with torch.no_grad()` internally already
        model=model,
        # The user text is provided via as `message.content`
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=35,
        context_size=gpt_config["context_length"],
        temperature=0.2, top_k=5, eos_id=50256
    )
    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)

    await chainlit.Message(
        # This returns the model response to the interface
        content=f"{response}",
    ).send()
