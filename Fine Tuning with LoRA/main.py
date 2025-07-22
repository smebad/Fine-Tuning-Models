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

# Loading the model with bitsandbytes configuration
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  quantization_config=bnb_config, # Apply bitsandbytes configuration
  device_map='auto', # Automatically map model to available devices
  trust_remote_code=True # Trust remote code for model loading
)

# Defining the LoRA configuration as per the original LoRA paper
lora_config = LoraConfig(
  r = 8, # Rank of the LoRA layers
  lora_alpha = 16, # Scaling factor for LoRA
  lora_dropout = 0.05, # Dropout rate for LoRA layers so they don't overfit
  target_modules = ['q_proj', 'v_proj'], # Target modules for LoRA
  bias = None, # No bias in LoRA layers
  task_type = TaskType.CAUSAL_LM # Task type for causal language modeling
)