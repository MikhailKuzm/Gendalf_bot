from datasets import Dataset
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd

# Указываем устройство CUDA, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры модели
model_checkpoint = "UrukHan/t5-russian-spell"
MAX_INPUT = 256

# Создаем папки для сохранения модели и токенизатора
os.makedirs("model", exist_ok=True)
os.makedirs("tokenizer", exist_ok=True)

# Загружаем токенизатор и модель
tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

# Читаем данные
data_df = pd.read_csv("Gandalf_Quotes.csv")
 
# Разделяем данные
train_df, val_df = train_test_split(data_df, test_size=0.1, random_state=42)
# Добавляем "Вопрос Гендальфу" к каждому вопросу перед токенизацией
train_df["Questions"] = train_df["Questions"].apply(lambda q: "Вопрос Гендальфу: " + q)
val_df["Questions"] = val_df["Questions"].apply(lambda q: "Вопрос Гендальфу: " + q)
# Функция токенизации
def tokenize_function(examples):
    model_inputs = tokenizer(examples["Questions"], padding="max_length", truncation=True, max_length=MAX_INPUT)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Answers"], padding="max_length", truncation=True, max_length=MAX_INPUT)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Создаём `Dataset`
train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
eval_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

# Настройки обучения
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Оценка на валидации после каждой эпохи
    save_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
    metric_for_best_model="eval_loss",  # Ориентируемся на минимальный loss
    load_best_model_at_end=True,  # Загружает лучшую модель по валидации
    greater_is_better=False,  # Чем меньше loss, тем лучше
    logging_dir="./logs",
    predict_with_generate=True,  # Используем генерацию при валидации
    fp16=True if torch.cuda.is_available() else False,
)

# Data collator для Seq2Seq моделей
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Создаём `Seq2SeqTrainer`
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Обучаем модель
trainer.train()

# Сохраняем модель и токенизатор
model.save_pretrained("model")
tokenizer.save_pretrained("tokenizer")