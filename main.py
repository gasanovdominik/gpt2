from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch

# 1. Загрузка и подготовка данных
data = {"text": [
    "Вопрос: Как сбросить пароль? Ответ: Перейдите в настройки и нажмите 'Забыли пароль'.",
    "Вопрос: Как обновить приложение? Ответ: Откройте магазин приложений и нажмите 'Обновить'.",
    "Вопрос: Почему не работает интернет? Ответ: Проверьте соединение и перезагрузите маршрутизатор."
]}

dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2)

# 2. Загрузка токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 3. Настройка обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 4. Запуск обучения
trainer.train()

# 5. Тестирование модели
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Пример тестирования
print(generate_response("Вопрос: Как включить Wi-Fi? Ответ:"))
