#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Запись речевых аудиовизуальных данных
"""

import os
import sys

PATH_TO_SOURCE = os.path.abspath(os.path.dirname(__file__))
PATH_TO_ROOT = os.path.join(PATH_TO_SOURCE, "..", "..")

sys.path.insert(0, os.path.abspath(PATH_TO_ROOT))

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings

for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

from dataclasses import dataclass  # Класс данных

import logging  # Логирование Функции создающие итераторы для эффективного цикла

from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.middleware.proxy_fix import ProxyFix

# Типы данных
from typing import Dict, Union, Any
from types import ModuleType

# Персональные
import openav  # Библиотека в целом
from openav.modules.trml.shell import Shell  # Работа с Shell
from openav.modules.lab.build import Run  # Сборка библиотеки
from openav import rsrs  # Ресурсы библиотеки

from openav.modules.core.logging import ARG_PATH_TO_LOGS

app = Flask(
    __name__,
    template_folder="../modules/dataset_recording/templates",
    static_folder="../modules/dataset_recording/static",
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

global video_path, output_dir
global processing_finished

processing_finished = False
video_path = "temp_dir/video.mp4"
output_dir = "cuted"


def read_questions_from_csv():
    questions = []
    # with open("questions.csv", mode="r", newline="", encoding="utf-8") as file:
    # csv_reader = csv.DictReader(file)
    # for row in csv_reader:
    #     question_number = int(row["QuestionNumber"])
    #     question_text = row["QuestionText"]
    #     disable_time = row["Disable_time"]
    #     questions.append(
    #         {"QuestionNumber": question_number, "QuestionText": question_text, "Disable_time": disable_time}
    #     )
    return questions


@app.route("/get_questions", methods=["GET"])
def get_questions():
    """Получение списка вопросов

    GET /get_questions

    Возвращает список вопросов в json формате
    """
    questions = read_questions_from_csv()
    return jsonify(questions=questions)


# Configure the upload folder for recorded videos
app.config["UPLOAD_FOLDER"] = "/Users/dl/GitHub/OpenAV/openav/modules/dataset_recording/static/recorded-video"

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


@app.route("/")
def index():
    """Отображение основной страницы записи

    GET /

    Отображает статику страницы записи. Является точкой входа в приложение
    """
    global processing_finished
    processing_finished = False
    return render_template("index_home.html")


@app.route("/store_timing_data", methods=["POST"])
def store_timing_data():
    """Сохранение файла временных отметок записи

    POST /store_timing_data

    Сохраняет файл временных отметок записи, который предается в виде json в теле запроса
    """
    if request.method == "POST":
        data = request.json
        if os.path.exists("timing_data.txt"):
            os.remove("timing_data.txt")
        with open("timing_data.txt", "w", encoding="utf-8") as file:
            for item in data:
                file.write(f"Question: {item['question']}, Timestamp: {item['timestamp']}\n")

            file.close()
        return "Data stored successfully"


@app.route("/upload", methods=["POST"])
def upload():
    """Сохранение файла записи

    POST /upload

    Сохраняет файл записи, который предается в виде файла видео в теле запроса
    """
    global processing_finished
    if "video" in request.files:
        video_file = request.files["video"]
        if video_file.filename != "":
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename) + ".webm"
            print(video_path)
            # video_file.save(video_path)
            if not os.path.exists("temp_dir"):
                os.makedirs("temp_dir")
            video_file.save("temp_dir/video.webm")
            # convert_webm_to_mp4('temp_dir/video.webm', 'temp_dir/video.mp4')
            print("Done")
            processing_finished = True

            return "DONE"

    return "No video data received."


@app.route("/download_processed_video")
def download_processed_video():
    """Выгрузка файла записи

    GET /download_processed_video

    Скачивает файл с записью в формате webm
    """
    processed_video_path = "temp_dir/video.webm"
    return send_file(processed_video_path, as_attachment=True)


@app.route("/download_timing_data")
def download_timing_data():
    """Выгрузка файла временных отметок записи

    GET /download_processed_video

    Скачивает файл временных отметок записи в формате txt
    """
    timing_data_path = "timing_data.txt"  # Path to the file
    # Set cache-control headers to prevent caching
    response = send_file(timing_data_path, as_attachment=True, mimetype="text/plain")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class MessagesRecorder(Run):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._description: str = self._("Запись речевых аудиовизуальных данных")
        self._description_time: str = "{}" * 2 + self._description + self._em + "{}"

        self._check_config_file_valid = self._("Проверка данных на валидность") + self._em


# ######################################################################################################################
# Выполняем только в том случае, если файл запущен сам по себе
# ######################################################################################################################
@dataclass
class RunRecorder(MessagesRecorder):
    """Класс для записи речевых аудиовизуальных данных"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._all_layer_in_yaml = 3  # Общее количество настроек в конфигурационном файле

        #  Регистратор логирования с указанным именем
        self._logger_run_train: logging.Logger = logging.getLogger(__class__.__name__)

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    def _build_args(self, description: str, conv_to_dict: bool = True, out=True) -> Dict[str, Any]:
        """
        Args:
            description (str): Описание парсера командной строки
            conv_to_dict (bool): Преобразование списка аргументов командной строки в словарь
            out (bool): Печатать процесс выполнения

        Returns:
            Dict[str, Any]: Словарь со списком аргументов командной стройки
        """

        # Выполнение функции из суперкласса
        super().build_args(description=description, conv_to_dict=False, out=out)

        if self._ap is None:
            return {}

        # Добавление аргументов в парсер командной строки
        self._ap.add_argument(
            "--config", required=True, metavar=self._("ФАЙЛ"), help=self._("Путь к конфигурационному файлу")
        )

        self._ap.add_argument(
            ARG_PATH_TO_LOGS,
            required=False,
            metavar=self._("ФАЙЛ"),
            help=self._("Путь к директории для сохранения LOG файлов"),
        )

        self._ap.add_argument(
            "--automatic_update",
            action="store_true",
            help=self._(
                "Автоматическая проверка конфигурационного файла в момент работы программы (работает при заданном"
            )
            + " --config",
        )
        self._ap.add_argument(
            "--no_clear_shell", action="store_false", help=self._("Не очищать консоль перед выполнением")
        )

        # Преобразование списка аргументов командной строки в словарь
        if conv_to_dict is True:
            args, _ = self._ap.parse_known_args()
            return vars(args)  # Преобразование списка аргументов командной строки в словарь

    def _valid_yaml_config(self, config: Dict[str, Union[str, bool, int, float]], out: bool = True) -> bool:
        """Проверка настроек JSON на валидность

        Args:
            config (Dict[str, Union[str, bool, int, float]]): Словарь из JSON файла
            out (bool): Печатать процесс выполнения

        Returns:
             bool: **True** если конфигурационный файл валиден, в обратном случае **False**
        """

        # Проверка аргументов
        if type(config) is not dict or type(out) is not bool:
            try:
                raise TypeError
            except TypeError:
                self.inv_args(__class__.__name__, self._valid_yaml_config.__name__, out=out)
                return False

        # Конфигурационный файл пуст
        if not config:
            try:
                raise TypeError
            except TypeError:
                self.message_error(self._config_empty, space=self._space, out=out)
                return False

        # Вывод сообщения
        self.message_info(self._check_config_file_valid, space=self._space, out=out)

        curr_valid_layer = 0  # Валидное количество разделов

        # Проход по всем разделам конфигурационного файла
        for key, val in config.items():
            # 1. Скрытие метаданных
            # 2. Скрытие версий установленных библиотек
            if key == "hide_metadata" or key == "hide_libs_vers":
                # Проверка значения
                if type(val) is not bool:
                    continue

                curr_valid_layer += 1

            # Словарь
            if key == "dictionary":
                # Проверка значения
                if type(val) is not list or len(val) == 0:
                    continue

                # Проход по всем классам
                for v in val:
                    # Проверка значения
                    if type(v) is not str or not v:
                        continue

                curr_valid_layer += 1

        # Сравнение общего количества ожидаемых настроек и валидных настроек в конфигурационном файле
        if self._all_layer_in_yaml != curr_valid_layer:
            try:
                raise TypeError
            except TypeError:
                self.message_error(self._invalid_file, space=self._space, out=out)
                return False

        return True  # Результат

    def _load_config_yaml(self, resources: ModuleType = rsrs, config="recorder.yaml", out: bool = True) -> bool:
        """Загрузка и проверка конфигурационного файла

        Args:
            resources (ModuleType): Модуль с ресурсами
            config (str): Конфигурационный файл
            out (bool): Печатать процесс выполнения

        Returns:
             bool: **True** если конфигурационный файл загружен и валиден, в обратном случае **False**
        """

        # Проверка аргументов
        if not isinstance(resources, ModuleType) or type(config) is not str or not config or type(out) is not bool:
            try:
                raise TypeError
            except TypeError:
                self.inv_args(__class__.__name__, self._load_config_yaml.__name__, out=out)
                return False

        # Конфигурационный файл передан
        if self._args["config"] is not None:
            config_yaml = self.load_yaml(self._args["config"], False, out)  # Загрузка YAML файла
        else:
            config_yaml = self.load_yaml_resources(resources, config, out)  # Загрузка YAML файла из ресурсов модуля

        # Конфигурационный файл не загружен
        if not config_yaml:
            return False

        # Проверка конфигурационного файла на валидность
        res_valid_yaml_config = self._valid_yaml_config(config_yaml, out)

        # Конфигурационный файл не валидный
        if res_valid_yaml_config is False:
            return False

        # Проход по всем разделам конфигурационного файла
        for k, v in config_yaml.items():
            self._args[k] = v  # Добавление значения из конфигурационного файла в словарь аргументов командной строки

        return True

    # ------------------------------------------------------------------------------------------------------------------
    #  Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def run(self, metadata: ModuleType = openav, resources: ModuleType = rsrs, out: bool = True) -> bool:
        """Запуск записи речевых аудиовизуальных данных

        Args:
            metadata (ModuleType): Модуль из которого необходимо извлечь информацию
            resources (ModuleType): Модуль с ресурсами
            out (bool): Печатать процесс выполнения

        Returns:
             bool: **True** если процесс записи речевых аудиовизуальных данных произведен успешно,
                   в обратном случае **False**
        """

        # Проверка аргументов
        if not isinstance(metadata, ModuleType) or not isinstance(resources, ModuleType) or type(out) is not bool:
            try:
                raise TypeError
            except TypeError:
                self.inv_args(__class__.__name__, self.run.__name__, out=out)
                return False

        self._args = self._build_args(self._description)  # Построение аргументов командной строки
        if len(self._args) == 0:
            return False

        # Очистка консоли перед выполнением
        if self.clear_shell(cls=self._args["no_clear_shell"], out=True) is False:
            return False

        # Вывод сообщения
        if out is True:
            # Приветствие
            Shell.add_line()  # Добавление линии во весь экран
            print(self._description_time.format(self.text_bold, self.color_blue, self.text_end))
            self._logger_run_train.info(self._description)
            Shell.add_line()  # Добавление линии во весь экран

        # Загрузка и проверка конфигурационного файла
        if self._load_config_yaml(resources, out=out) is False:
            return False

        # Вывод сообщения
        if out is True:
            Shell.add_line()  # Добавление линии во весь экран

        # Информация об библиотеке
        if self._args["hide_metadata"] is False and out is True:
            self.message_metadata_info(out=out)
            Shell.add_line()  # Добавление линии во весь экран

        # Версии установленных библиотек
        if self._args["hide_libs_vers"] is False and out is True:
            self.libs_vers(out=out)
            Shell.add_line()  # Добавление линии во весь экран

        app.run(debug=False)

        return True


def main():
    # Запуск процесса записи
    recorder = RunRecorder(lang="ru", path_to_logs="./openav/logs")
    recorder.run(out=True)


if __name__ == "__main__":
    main()
