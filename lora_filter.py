import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = AutoModelForCausalLM.from_pretrained(
    'checkpoints/galactica-base',
    
    device_map = 'auto',
    load_in_8bit=False,
    torch_dtype = torch.float16
)

tokenizer = AutoTokenizer.from_pretrained('checkpoints/galactica-base')
tokenizer.pad_token = '<pad>'

for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        # stablize small parameters
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r = 16,
    lora_alpha=32,
    target_modules = ['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias = 'none',
    task_type = 'CAUSAL_LM'
)

model = get_peft_model(model, config)

print_trainable_parameters(model)

data = load_dataset('csv', data_files='lora_filter.csv')
data = data.map(lambda samples: tokenizer(samples['document'], max_length = 1024), batched = True).shuffle(seed = 42)

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=2,
        warmup_steps=1000, 
        max_steps=20000, 
        save_steps = 2000,
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='checkpoints/galactica-filter'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()




