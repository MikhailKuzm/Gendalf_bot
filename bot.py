import telebot 
import random
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM
import torch
import os
tokenizer = T5TokenizerFast.from_pretrained('tokenizer')
model = AutoModelForSeq2SeqLM.from_pretrained('model').to('cpu')
bot = telebot.TeleBot('API KEY')
# Папка с изображениями Гэндальфа
images_folder = "images_gendalf"

# Функция для выбора случайного изображения Гэндальфа
def get_random_gandalf_image():
    if not os.path.exists(images_folder):
        return None  # Если папка отсутствует, не отправляем изображение

    images = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    if not images:
        return None  # Если изображений нет, пропускаем

    return os.path.join(images_folder, random.choice(images))

# Функция для генерации ответа Гэндальфа
def generate_gandalf_response(input_text):
    question = tokenizer('Вопрос Гендальфу: ' + input_text.lower(), return_tensors='pt')['input_ids'].to('cpu')
    with torch.no_grad():
        answer = model.generate(
            question, 
            max_length=80, 
            num_return_sequences=1, 
            do_sample=True, 
            num_beams=4,
            top_p=0.8, 
            temperature=1.3
        )
    return tokenizer.decode(answer[0], skip_special_tokens=True)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    input_string = message.text
    print(input_string)

    # Генерация ответа
    answer = generate_gandalf_response(input_string)
    print(answer)
    # Выбор случайного изображения
    image_path = get_random_gandalf_image()

    # Отправка текста и изображения
    bot.reply_to(message, answer)
    if image_path:
        with open(image_path, 'rb') as image:
            bot.send_photo(message.chat.id, image)

bot.polling()