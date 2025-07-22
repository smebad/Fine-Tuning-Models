import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset

from peft import LoraConfig, get_peft_model, TaskType

# Loading the model and tokenizer
model_name = 'TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) # Load the tokenizer with remote code trust

# bitsandbytes configuration
bnb_config = BitsAndBytesConfig(
  load_in_4bit=True, # Load model in 4-bit precision
  bnb_4bit_quant_type='nf4', # Use the NF4 quantization type
  bnb_4bit_compute_dtype=torch.float16 # Use float16 for computation
)