from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

###
# Questions:
# 1  What is missing in the code? 
# 2. Set the context length to 64 tokens
# 3. Set the batch size to 8
# 4. Set the padding side appropriately for training
# 5. Set the learning rate to 2e-5
# 6. What optimizer is used?
# 7. Change the default optimizer
# 8. Add a pad token to the tokenizer 
# 9. What datacollator is used?
# 10. Add a collator with dynamic padding
# 11. Add a custom trainer and set the optimizer to AdamW and LinearLR scheduler
# 12. Add a custom sampler to the custom trainer



# Load pre-trained model and tokenizer
model_name = "openai-community/gpt2"  # Replace with the model you want to fine-tune
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
)

model = AutoModelForCausalLM.from_pretrained(model_name)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")