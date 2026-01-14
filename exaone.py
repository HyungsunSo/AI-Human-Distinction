import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from utils import load_config
from dataset import DataSet

config = load_config("config.yaml")
model_name = config["model"]["model_name"]
device = "cuda" if torch.cuda.is_available() else "cpu"

def chat_template(example, tokenizer):
    label = "AI" if example["label"] == 1 else "Human"
    messages = [
            {"role": "system", "content": "당신은 텍스트 판별기입니다. 주어진 문장을 모두 읽어본 뒤, AI가 작성한 글인지 사람이 작성한 글인지 판별합니다. AI가 작성한 글일 경우 'AI', 사람이 작성한 글일 경우 'Human'만 출력합니다."},
            {"role": "user", "content": f"다음 글을 분류하세요: \n\n{example['full_text']}"},
            {"role": "assistant", "content": label},
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = False,
    )
    return {"text": text}
dataset = DataSet()
train_dataset, valid_dataset = dataset.get_tokenized_dataset()
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast = False,
    trust_remote_code = True,
)

train_sft = train_dataset.map(
    lambda ex: chat_template(ex, tokenizer),
    remove_columns = train_dataset.column_names,
)
valid_sft = valid_dataset.map(
    lambda ex: chat_template(ex, tokenizer),
    remove_columns = valid_dataset.column_names,
)

# bnb_cfg = BitsAndBytesConfig(
#     load_in_4bit = True,
#     bnb_4bit_quant_type = "nf4",
#     bnb_4bit_compute_dtype = torch.bfloat16,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config = bnb_cfg,
    device_map = device,
    trust_remote_code = True,
)

sft_cfg = SFTConfig(
    output_dir = config["sft_config"]["output_dir"],
    per_device_train_batch_size = config["sft_config"]["per_device_batch_size"],
    gradient_accumulation_steps = config["sft_config"]["gradient_accumulation_steps"],
    learning_rate = config["sft_config"]["lr"],
    num_train_epochs = config["sft_config"]["epochs"],
    max_seq_length = config["model"]["h_param"]["max_length"],
    logging_steps = config["sft_config"]["logging_steps"],
    save_steps = config["sft_config"]["save_steps"],
)

lora = LoraConfig(
    config["lora"]["r"],
    config["lora"]["lora_alpha"],
    config["lora"]["lora_dropout"],
    config["lora"]["bias"],
    config["lora"]["task_type"],
    config["lora"]["target_modules"],
)


trainer = SFTTrainer(
    model = model_name,
    tokenizer = tokenizer,
    train_dataset = train_sft,
    peft_config = lora,
    args = sft_cfg,
    dataset_text_field = "full_text"
)

# # trainer.train()