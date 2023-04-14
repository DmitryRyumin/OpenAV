#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Детектирование речевой активности в аудиовизуальном сигнале
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

import logging  # Логирование
import itertools  # Функции создающие итераторы для эффективного цикла

# Типы данных
from typing import Dict, Union, Any
from types import ModuleType

# Персональные
import openav  # Библиотека в целом
from openav.modules.trml.shell import Shell  # Работа с Shell
from openav.modules.lab.build import Run  # Сборка библиотеки
from openav import rsrs  # Ресурсы библиотеки

from openav.modules.core.logging import ARG_PATH_TO_LOGS
from openav.modules.lab.audio import (
    TYPES_ENCODE,
    PRESETS_CRF_ENCODE,
    SR_INPUT_TYPES,
    SAMPLING_RATE_VAD,
    WINDOW_SIZE_SAMPLES_VAD,
)


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class MessagesVAD(Run):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._description: str = self._("Детектирование речевой активности в аудиовизуальном сигнале")
        self._description_time: str = "{}" * 2 + self._description + self._em + "{}"

        self._check_config_file_valid = self._("Проверка данных на валидность") + self._em


# ######################################################################################################################
# Выполняем только в том случае, если файл запущен сам по себе
# ######################################################################################################################
@dataclass
class RunVAD(MessagesVAD):
    """Класс для детектирования речевой активности в аудиовизуальном сигнале"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._all_layer_in_yaml = 20  # Общее количество настроек в конфигурационном файле

        #  Регистратор логирования с указанным именем
        self._logger_runvad: logging.Logger = logging.getLogger(__class__.__name__)

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
            + " --config)",
        )
        self._ap.add_argument(
            "--no_clear_shell", action="store_false", help=self._("Не очищать консоль перед выполнением")
        )

        # Преобразование списка аргументов командной строки в словарь
        if conv_to_dict is True:
            args, _ = self._ap.parse_known_args()
            return vars(args)  # Преобразование списка аргументов командной строки в словарь

    def _valid_yaml_config(self, config: Dict[str, Union[str, bool, int, float]], out: bool = True) -> bool:
        """Проверка настроек YAML на валидность

        Args:
            config (Dict[str, Union[str, bool, int, float]]): Словарь из YAML файла
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
        for (key, val) in config.items():
            # 1. Скрытие метаданных
            # 2. Скрытие версий установленных библиотек
            # 3. Принудительная загрузка модели из сети
            # 4. Очистка директории для сохранения фрагментов аудиовизуального сигнала
            if key == "hide_metadata" or key == "hide_libs_vers" or key == "force_reload" or key == "clear_dirvad":
                # Проверка значения
                if type(val) is not bool:
                    continue

                curr_valid_layer += 1

            # Тип кодирования
            if key == "type_encode":
                # Проверка значения
                if type(val) is not str or (val in TYPES_ENCODE) is False:
                    continue

                curr_valid_layer += 1

            # Качество кодирования
            if key == "crf_value":
                # Проверка значения
                if type(val) is not int or not (0 <= val <= 51):
                    continue

                curr_valid_layer += 1

            # Скорость кодирования и сжатия
            if key == "presets_crf_encode":
                # Проверка значения
                if type(val) is not str or (val in PRESETS_CRF_ENCODE) is False:
                    continue

                curr_valid_layer += 1

            # Тип файлов для распознавания речи
            if key == "sr_input_type":
                # Проверка значения
                if type(val) is not str or (val in SR_INPUT_TYPES) is False:
                    continue

                curr_valid_layer += 1

            # 1. Путь к директории для сохранения моделей
            # 2. Путь к директории набора данных
            # 3. Путь к директории набора данных состоящего из фрагментов аудиовизуального сигнала
            if key == "path_to_save_models" or key == "path_to_dataset" or key == "path_to_dataset_vad":
                # Проверка значения
                if type(val) is not str or not val:
                    continue

                curr_valid_layer += 1

            # Глубина иерархии для получения данных
            if key == "depth":
                # Проверка значения
                if type(val) is not int or not (1 <= val <= 10):
                    continue

                curr_valid_layer += 1

            # Частота дискретизации
            if key == "sampling_rate":
                # Проверка значения
                if type(val) is not int or (val in SAMPLING_RATE_VAD) is False:
                    continue

                curr_valid_layer += 1

            # Порог вероятности речи
            if key == "threshold":
                # Проверка значения
                if type(val) is not float or not (0.0 <= val <= 1.0):
                    continue

                curr_valid_layer += 1

            # 1. Минимальная длительность речевого фрагмента в миллисекундах
            # 2. Минимальная длительность тишины в выборках между отдельными речевыми фрагментами
            # 3. Внутренние отступы для итоговых речевых фрагментов
            if key == "min_speech_duration_ms" or key == "min_silence_duration_ms" or key == "speech_pad_ms":
                # Проверка значения
                if type(val) is not int or not 1 <= val:
                    continue

                curr_valid_layer += 1

            # Количество выборок в каждом окне
            if key == "window_size_samples":
                # Проверка значения
                if (
                    type(val) is not int
                    or (val in set(itertools.chain.from_iterable(WINDOW_SIZE_SAMPLES_VAD.values()))) is False
                ):
                    continue

                curr_valid_layer += 1

            # Названия директорий для видео и аудио
            if key == "dir_va_names":
                all_layer_2 = 2  # Общее количество подразделов в текущем разделе
                curr_valid_layer_2 = 0  # Валидное количество подразделов в текущем разделе

                # Проверка значения
                if type(val) is not dict or len(val) == 0:
                    continue

                # Проход по всем подразделам текущего раздела
                for (k, v) in val.items():
                    # Проверка значения
                    if type(v) is not str or not v:
                        continue

                    # 1. Директории с разделенными видеофрагментами
                    # 2. Директории с разделенными аудиофрагментами
                    if k == "video" or k == "audio":
                        curr_valid_layer_2 += 1

                if all_layer_2 == curr_valid_layer_2:
                    curr_valid_layer += 1

            # Расширения искомых файлов
            if key == "ext_search_files":
                curr_valid_layer_2 = 0  # Валидное количество подразделов в текущем разделе

                # Проверка значения
                if type(val) is not list or len(val) == 0:
                    continue

                # Проход по всем подразделам текущего раздела
                for v in val:
                    # Проверка значения
                    if type(v) is not str or not v:
                        continue

                    curr_valid_layer_2 += 1

                if curr_valid_layer_2 > 0:
                    curr_valid_layer += 1

        # Сравнение общего количества ожидаемых настроек и валидных настроек в конфигурационном файле
        if self._all_layer_in_yaml != curr_valid_layer:
            try:
                raise TypeError
            except TypeError:
                self.message_error(self._invalid_file, space=self._space, out=out)
                return False

        return True  # Результат

    def _load_config_yaml(self, resources: ModuleType = rsrs, config="vad.yaml", out: bool = True) -> bool:
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
        for (k, v) in config_yaml.items():
            self._args[k] = v  # Добавление значения из конфигурационного файла в словарь аргументов командной строки

        return True

    # ------------------------------------------------------------------------------------------------------------------
    #  Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def run(self, metadata: ModuleType = openav, resources: ModuleType = rsrs, out: bool = True) -> bool:
        """Запуск детектирования речевой активности в аудиовизуальном сигнале

        Args:
            metadata (ModuleType): Модуль из которого необходимо извлечь информацию
            resources (ModuleType): Модуль с ресурсами
            out (bool): Печатать процесс выполнения

        Returns:
             bool: **True** если детектирование речевой активности в аудиовизуальном сигнале произведено успешно,
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
            self._logger_runvad.info(self._description)
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

        self.path_to_save_models = self._args["path_to_save_models"]  # Путь к директории для сохранения моделей
        self.path_to_dataset = self._args["path_to_dataset"]  # Путь к директории набора данных
        # Путь к директории набора данных состоящего из фрагментов аудиовизуального сигнала
        self.path_to_dataset_vad = self._args["path_to_dataset_vad"]
        self.dir_va_names = self._args["dir_va_names"]  # Названия директорий для видео и аудио
        self.ext_search_files = self._args["ext_search_files"]  # Расширения искомых файлов

        # print()
        # print(self._args["threshold"])
        # print(self._args["min_speech_duration_ms"])
        # print(self._args["min_silence_duration_ms"])
        # print(self._args["window_size_samples"])
        # print(self._args["speech_pad_ms"])
        # return

        self.vad(
            depth=self._args["depth"],  # Глубина иерархии для получения данных
            type_encode=self._args["type_encode"],  # Тип кодирования
            crf_value=self._args["crf_value"],  # Качество кодирования
            presets_crf_encode=self._args["presets_crf_encode"],  # Скорость кодирования и сжатия
            sr_input_type=self._args["sr_input_type"],  # Тип файлов для распознавания речи
            sampling_rate=self._args["sampling_rate"],  # Частота дискретизации
            threshold=self._args["threshold"],  # Порог вероятности речи
            # Минимальная длительность речевого фрагмента в миллисекундах
            min_speech_duration_ms=self._args["min_speech_duration_ms"],
            # Минимальная длительность тишины в выборках между отдельными речевыми фрагментами
            min_silence_duration_ms=self._args["min_silence_duration_ms"],
            window_size_samples=self._args["window_size_samples"],  # Количество выборок в каждом окне
            # Внутренние отступы для итоговых речевых фрагментов
            speech_pad_ms=self._args["speech_pad_ms"],
            force_reload=self._args["force_reload"],  # Принудительная загрузка модели из сети
            # Очистка директории для сохранения фрагментов аудиовизуального сигнала
            clear_dirvad=self._args["clear_dirvad"],
            out=out,
        )

        return True


def main():
    # Запуск детектирования речевой активности в аудиовизуальном сигнале
    vad = RunVAD(lang="ru", path_to_logs="./openav/logs")
    vad.run(out=True)


if __name__ == "__main__":
    main()
