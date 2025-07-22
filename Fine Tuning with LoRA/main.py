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

# Applying the LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Loading the GSM8K dataset from the Hugging Face Hub
data = load_dataset('openai/gsm8k', 'main', split = 'train[:200]')

# Tokenization function to prepare the dataset for training
def tokenize_function(batch):
    texts = [f"Instruction:\n{batch['question']}\n### Response: \n{batch['answer']}" # Format the input text 
             for instruction, out in zip(batch['question'], batch['answer']) # Combine question and answer
    ]
    
    tokens = tokenizer(
        texts, # Tokenize the formatted texts
        padding = 'max_length', # Pad to max length
        truncation = True, # Truncate if necessary
        return_tensors = 'pt' # Return PyTorch tensors
    )

    tokens['labels'] = tokens['input_ids'].clone() # Clone input IDs for labels

    return tokens

# Tokenizing the dataset using the defined function
tokenized_data = data.map(tokenize_function, batched=True, remove_columns=data.column_names)

# Setting up training arguments
training_args = TrainingArguments(
    output_dir = './lora-tinylama-math', # Directory to save the model
    per_device_train_batch_size= 1, # Batch size for training
    gradient_accumulation_steps = 4, # Gradient accumulation steps
    learning_rate = 1e-3, # Learning rate for training
    num_train_epochs = 30, # Number of training epochs
    fp16 = True, # Use mixed precision training
    logging_steps = 10, # Log every 10 steps
    save_strategy = 'epoch', # Save the model at the end of each epoch
    report_to = 'none', # Disable reporting to external services
    remove_unused_columns = False, # Do not remove unused columns
    label_names = ['labels'], # Specify label names for training
)

# Initializing the Trainer with the model, training arguments, and tokenized dataset
trainer = Trainer(
    model = model, # The model to train
    args = training_args, # Training arguments
    train_dataset = tokenized_data, # Tokenized dataset for training
    processing_class = tokenizer, # Tokenizer for processing inputs
)