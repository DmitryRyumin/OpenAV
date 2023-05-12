#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тестирование разработанных модулей
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
    VOSK_SUPPORTED_LANGUAGES,
    VOSK_SUPPORTED_DICTS,
)


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class MessageTesting(Run):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._description: str = self._("Тестирование разработанных модулей")
        self._description_time: str = "{}" * 2 + self._description + self._em + "{}"

        self._check_config_file_valid = self._("Проверка данных на валидность") + self._em


# ######################################################################################################################
# Выполняем только в том случае, если файл запущен сам по себе
# ######################################################################################################################
@dataclass
class RunTesting(MessageTesting):
    """Класс для тестирования разработанных модулей"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        #  Регистратор логирования с указанным именем
        self._logger_run_vosk_sr: logging.Logger = logging.getLogger(__class__.__name__)
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
        """Запуск тестирования разработанных модулей

        Args:
            metadata (ModuleType): Модуль из которого необходимо извлечь информацию
            resources (ModuleType): Модуль с ресурсами
            out (bool): Печатать процесс выполнения

        Returns:
             bool: **True** если тестирование разработанных модулей произведено успешно,
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
            self._logger_run_vosk_sr.info(self._description)
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

        # Проверка модуля загрузки данных
        self.download_data(out=out)

        # Проверка модуля детектирования речевой активности
        self.vosk_sr(
            depth=self._args["depth"],  # Глубина иерархии для получения данных
            type_encode=self._args["type_encode"],  # Тип кодирования
            crf_value=self._args["crf_value"],  # Качество кодирования
            presets_crf_encode=self._args["presets_crf_encode"],  # Скорость кодирования и сжатия
            new_name=self._args["folder_name_unzip"],  # Имя директории для разархивирования
            # Внутренний левый отступ для итоговых речевых фрагментов
            speech_left_pad_ms=self._args["speech_left_pad_ms"],
            # Внутренний правый отступ для итоговых речевых фрагментов
            speech_right_pad_ms=self._args["speech_right_pad_ms"],
            force_reload=self._args["force_reload"],  # Принудительная загрузка модели из сети
            # Очистка директории для сохранения фрагментов аудиовизуального сигнала
            clear_dirvosk_sr=self._args["clear_dirvosk_sr"],
            out=out,
        )

        # Проверка модуля предобработки речевых аудиоданных
        self.preprocessing_audio(
            sample_rate=self._args["sample_rate"],  # Частота дискретизации аудиосигнала
            n_fft=self._args["n_fft"],  # Размер параметра FFT
            hop_length=self._args["hop_length"],  # Длина перехода между окнами STFT
            n_mels=self._args["n_mels"],  # Количество фильтроблоков mel
            power=self._args["power"],  # Показатель степени магнитудной спектрограммы
            pad_mode=self._args["pad_mode"],  # Управление оступами
            center=self._args["center"],  # Включение установки отступов с обеих сторон относительно центра
            # Очистка директории для сохранения фрагментов аудиовизуального сигнала
            clear_dirvosk_sr=self._args["clear_dirvosk_sr"],
            out=out,
        )

        # Проверка модуля предобработки речевых видеоданных
        self.preprocessing_video(
            depth=self._args["depth"],  # Глубина иерархии для получения данных
            width=self._args["width"],  # Ширина кадра с найденной областью губ
            height=self._args["height"],  # Высота кадра с найденной областью губ
            color_mode=self._args["color_mode"],  # Цветовая гамма конечного изображения
            # Очистка директории для сохранения фрагментов аудиовизуального сигнала
            clear_dirvosk_sr=self._args["clear_dirvosk_sr"],
            out=out,
        )

        return True


def main():
    # Запуск тестирования разработанных модулей
    vad = RunTesting(lang="ru", path_to_logs="./openav/logs")
    vad.run(out=True)


if __name__ == "__main__":
    main()
