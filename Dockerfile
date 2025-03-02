# Используем базовый образ с Python
FROM python:3.9

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir torch transformers telebot

# Указываем переменную окружения, чтобы не было буферизации вывода
ENV PYTHONUNBUFFERED=1

# Запускаем бота
CMD ["python", "bot.py"]