from torch.utils.data import Sampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

###
# Questions:
# 1 What is missing in the code?
# 2. Set the context length to 64 tokens
# 2. Set the batch size to 8
# 3. Set the learning rate to 2e-5
# 4. Set the optimizer to "adam"
# 5. Add a custom trainer and set the optimizer to AdamW




class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # Custom logic for sampling indices
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        # Use the custom sampler
        train_sampler = CustomSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
        )
        

    def create_optimizer_and_scheduler(self, num_training_steps):
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)
        self.lr_scheduler = LinearLR(
            self.optimizer, 0, num_training_steps, power=2)


# Load pre-trained model and tokenizer
model_name = "openai-community/gpt2"  # Replace with the model you want to fine-tune
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split dataset
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
)

# Define the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name)

optimizer = Adam(model.parameters(), lr=2e-5)
lr_scheduler = LinearLR(optimizer)


## custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)


# Default trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, lr_scheduler),
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")