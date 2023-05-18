from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import load_dataset
import evaluate

def preprocess_function(examples):
    return tokenizer(examples["document"], truncation=True, max_length = 1024)
tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-large-4096')
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
id2label = {0:'Neither', 1:'Support', 2:'Contradict'}
label2id = {'Neither': 0, 'Support': 1, 'Contradict': 2}


model = AutoModelForSequenceClassification.from_pretrained(
    'allenai/longformer-large-4096',
    num_labels = 3,
    id2label = id2label,
    label2id = label2id,
    torch_dtype = torch.float16,
    device_map = 'auto'
)


data = load_dataset('csv', data_files='./class_ft_data.csv')
data = data.map(preprocess_function, batched = True)
test_data = load_dataset('csv', data_files='./class_ft_test_data.csv')
test_data = test_data.map(preprocess_function, batched = True)



accuracy = evaluate.load('accuracy')


def compute_metrices(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return accuracy.compute(predictions=predictions, reference = labels)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps = 100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=test_data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrices,
    
)

trainer.train()