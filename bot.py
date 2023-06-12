import os
import telebot
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

TOKEN = os.environ.get("TOKEN")

ALLOWED_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm', '.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.webm']
Model = YOLO('yolov8n.pt')

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'Привет! Я бот для детектирования объектов на изображениях и видео. Можно отправить даже документом. \nДля уточнения поддерживаемых форматах напишите команду "/help"')

@bot.message_handler(commands=['help'])
def send_help(message):
    available_formats = "Доступные форматы для обработки:\n\n" \
                        "Видео: '.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.webm'\n" \
                        "Фото: '.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm'"

    bot.reply_to(message, available_formats)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    bot.reply_to(message, 'Отправьте мне файл и я помогу вам с детекцией.')

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        file_stream = bot.download_file(file_info.file_path)

        img = cv2.imdecode(np.frombuffer(file_stream, np.uint8), -1)
        res = Model(img)
        res_plotted = res[0].plot()

        image_bytes = BytesIO()
        Image.fromarray(res_plotted).save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        bot.send_photo(message.chat.id, photo=image_bytes)
        bot.reply_to(message, 'Сделяль')

    except Exception as e:
        bot.reply_to(message, 'Произошла ошибка при обработке фото. Пожалуйста, попробуйте еще раз.')

@bot.message_handler(content_types=['video'])
def handle_video(message):
    try:
        file_info = bot.get_file(message.video.file_id)
        file_extension = '.' + file_info.file_path.split('.')[-1]

        if file_extension.lower() not in ALLOWED_FORMATS:
            bot.reply_to(message, 'Неподдерживаемый формат файла. Пожалуйста, загрузите другое видео.')
            return

        bot.reply_to(message, 'Подождите пару минут, пока видео обрабатывается')

        file_stream = bot.download_file(file_info.file_path)

        # Сохраняем видео во временный файл
        video_path = 'video.MP4'
        with open(video_path, 'wb') as f:
            f.write(file_stream)

        cap = cv2.VideoCapture(video_path)

        # Пропорции видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # выход
        output_path = "/Users/gor/PycharmProjects/Detection_tg_bot/output.MP4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # покадровый проход

        while cap.isOpened():
            #
            success, frame = cap.read()

            if success:

                results = Model(frame)
                annotated_frame = results[0].plot()
                output_video.write(annotated_frame)

            else:
                break


        cap.release()
        output_video.release()

        # Отправляем видеофайл
        with open('output.MP4', 'rb') as f:
            bot.send_video(message.chat.id, f)
        bot.reply_to(message, 'Сделяль')

        # Удаляем временные файлы
        os.remove(video_path)
        os.remove(output_path)

    except Exception as e:
        bot.reply_to(message, 'Произошла ошибка при обработке видел. Пожалуйста, попробуйте другой файл.')

def handle_document(message):
    file_info = bot.get_file(message.document.file_id)
    file_extension = '.' + file_info.file_path.split('.')[-1]

    if file_extension.lower() in ALLOWED_FORMATS:
        # Обработка видео
        if file_extension.lower() in ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.webm']:
            handle_video(message)
        # Обработка фото
        elif file_extension.lower() in ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm']:
            handle_photo(message)
        else:
            bot.reply_to(message, 'Неподдерживаемый формат файла. Пожалуйста, загрузите другой документ.')
    else:
        bot.reply_to(message, 'Неподдерживаемый формат файла. Пожалуйста, загрузите другой документ.')

bot.polling(none_stop=True)
