#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Детектирование речевой активности в аудиовизуальном сигнале
"""

import os
import sys
PATH_TO_SOURCE = os.path.abspath(os.path.dirname(__file__))
PATH_TO_ROOT = os.path.join(PATH_TO_SOURCE, '..', '..')

sys.path.insert(0, os.path.abspath(PATH_TO_ROOT))

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings
for warn in [UserWarning, FutureWarning]: warnings.filterwarnings('ignore', category = warn)

from dataclasses import dataclass # Класс данных

# Типы данных
from typing import Dict, Union, Any
from types import ModuleType

# Персональные
import openav                               # Библиотека в целом
from openav.modules.trml.shell import Shell # Работа с Shell
from openav.modules.lab.build import Run    # Сборка библиотеки
from openav import rsrs                     # Ресурсы библиотеки

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
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._description: str = self._('Детектирование речевой активности в аудиовизуальном сигнале')
        self._description_time: str = '{}' * 2 + self._description + self._em + '{}'

        self._check_config_file_valid = self._('Проверка данных на валидность') + self._em

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
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._all_layer_in_json = 1 # Общее количество настроек в конфигурационном файле

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    def _build_args(self, description: str, conv_to_dict: bool = True, out = True) -> Dict[str, Any]:
        """
        Args:
            description (str): Описание парсера командной строки
            conv_to_dict (bool): Преобразование списка аргументов командной строки в словарь
            out (bool): Печатать процесс выполнения

        Returns:
            Dict[str, Any]: Словарь со списком аргументов командной стройки
        """

        # Выполнение функции из суперкласса
        super().build_args(description = description, conv_to_dict = False, out = out)

        # Добавление аргументов в парсер командной строки
        self._ap.add_argument('--config', metavar = self._('ФАЙЛ'),
                              help = self._('Путь к конфигурационному файлу'))

        self._ap.add_argument('--automatic_update', action = 'store_true',
                              help = self._('Автоматическая проверка конфигурационного файла в момент работы программы '
                                            '(работает при заданном') + ' --config')
        self._ap.add_argument('--no_clear_shell', action = 'store_false',
                              help = self._('Не очищать консоль перед выполнением'))

        # Преобразование списка аргументов командной строки в словарь
        if conv_to_dict is True:
            args, _ = self._ap.parse_known_args()
            return vars(args) # Преобразование списка аргументов командной строки в словарь

    # Проверка JSON файла настроек на валидность
    def _valid_json_config(self, config: Dict[str, Union[str, bool, int, float]], out: bool = True) -> bool:
        """Проверка настроек JSON на валидность

        Args:
            config (Dict[str, Union[str, bool, int, float]]): Словарь из JSON файла
            out (bool): Печатать процесс выполнения

        Returns:
             bool: **True** если конфигурационный файл валиден, в обратном случае **False**
        """

        # Проверка аргументов
        if type(config) is not dict or type(out) is not bool:
            self.inv_args(__class__.__name__, self._valid_json_config.__name__, out = out); return False

        # Конфигурационный файл пуст
        if not config: self.message_error(self._config_empty, space = self._space, out = out); return False

        # Вывод сообщения
        self.message_info(self._check_config_file_valid, space = self._space, out = out)

        curr_valid_layer = 0 # Валидное количество разделов

        # Проход по всем разделам конфигурационного файла
        for key, val in config.items():
            # 1. Скрытие метаданных
            if key == 'hide_metadata':
                # Проверка значения
                if type(val) is not bool: continue

                curr_valid_layer += 1

        # Сравнение общего количества ожидаемых настроек и валидных настроек в конфигурационном файле
        if self._all_layer_in_json != curr_valid_layer:
            self.message_error(self._invalid_file, space = self._space, out = out); return False

        return True # Результат

    def _load_config_json(self, resources: ModuleType = rsrs, config = 'vad.json', out: bool = True) -> bool:
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
            self.inv_args(__class__.__name__, self._load_config_json.__name__, out = out); return False

        # Конфигурационный файл передан
        if self._args['config'] is not None:
            config_json = self.load_json(self._args['config'], False, out) # Загрузка JSON файла
        else: config_json = self.load_json_resources(resources, config, out) # Загрузка JSON файла из ресурсов модуля

        # Конфигурационный файл не загружен
        if not config_json: return False

        # Проверка конфигурационного файла на валидность
        res_valid_json_config = self._valid_json_config(config_json, out)

        # Конфигурационный файл не валидный
        if res_valid_json_config is False: return False

        # Проход по всем разделам конфигурационного файла
        for k, v in config_json.items():
            self._args[k] = v # Добавление значения из конфигурационного файла в словарь аргументов командной строки

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
            self.inv_args(__class__.__name__, RunVAD.run.__name__, out = out); return False

        self._args = self._build_args(self._description) # Построение аргументов командной строки

        # Очистка консоли перед выполнением
        if self.clear_shell(cls = self._args['no_clear_shell'], out = True) is False: return False

        # Вывод сообщения
        if out is True:
            # Приветствие
            Shell.add_line() # Добавление линии во весь экран
            print(self._description_time.format(self.text_bold, self.color_blue, self.text_end))
            Shell.add_line() # Добавление линии во весь экран

        # Загрузка и проверка конфигурационного файла
        if self._load_config_json(resources, out = out) is False: return False

        # Вывод сообщения
        if out is True: Shell.add_line() # Добавление линии во весь экран

        # Запуск
        if self._args['hide_metadata'] is False and out is True:
            self.message_metadata_info(out)

            Shell.add_line() # Добавление линии во весь экран

        return True

def main():
    # Запуск детектирования речевой активности в аудиовизуальном сигнале
    vad = RunVAD(lang = 'ru')
    vad.run()


if __name__ == "__main__":
    main()
