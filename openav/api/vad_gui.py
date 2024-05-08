#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Графический интерфейс пользователя для детектирования речевой активности в аудиовизуальном сигнале
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

import importlib.resources as pkg_resources  # Работа с ресурсами внутри пакетов

from dataclasses import dataclass  # Класс данных

import logging  # Логирование

import streamlit as st  # Графический интерфейс пользователя
from PIL import Image

# Типы данных
from types import ModuleType

# Персональные
import openav  # Библиотека в целом
from openav.api.vad import RunVAD  # Сборка библиотеки

# Ресурсы библиотеки
from openav import rsrs
from openav.rsrs import favicon


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class MessagesVAD_GUI(RunVAD):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса


# ######################################################################################################################
# Выполняем только в том случае, если файл запущен сам по себе
# ######################################################################################################################
@dataclass
class RunVAD_GUI(MessagesVAD_GUI):
    """Класс для детектирования речевой активности в аудиовизуальном сигнале"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        #  Регистратор логирования с указанным именем
        self._logger_runvad_gui: logging.Logger = logging.getLogger(__class__.__name__)

        self.logger_gui = True  # Установка логирования GUI

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

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
            except Exception:
                self.inv_args(__class__.__name__, self.run.__name__, out=out)

                container = """<div style="
                    display:block; overflow:overlay; height:300px; padding:12px; border:1px solid #FFF
                ">"""

                for message in self._get_logger_messages_gui():
                    container += message

                container += "</div>"

                st.markdown(container, unsafe_allow_html=True)

                return False

        # # Ресурс с YAML файлом не найден
        # if pkg_resources.is_resource(module, path_to_file) is False:
        #     self.message_error(self._load_data_resources_not_found, space=self._space, out=out)
        #     return {}

        # # Открытие файла
        # with pkg_resources.open_text(module, path_to_file, encoding="utf-8", errors="strict") as yaml_data_file:
        #     try:
        #         config = yaml.load(yaml_data_file, Loader=yaml.FullLoader)
        #     except yaml.YAMLError:
        #         self.message_error(self._invalid_file, space=self._space, out=out)
        #         return {}

        # print(pkg_resources.is_resource(resources.favicon, "favicon.ico"))

        # with pkg_resources.open_text(
        #     resources, "/favicon/avicon.ico", encoding="utf-8", errors="strict"
        # ) as favicon_file:
        #     print(favicon_file.name)
        #     favicon_image = Image.open(favicon_file.name)

        #     # Настройки параметров страницы
        #     st.set_page_config(
        #         page_title=self._description,
        #         page_icon=favicon_image,
        #         layout="wide",
        #         initial_sidebar_state="expanded",
        #         menu_items={
        #             "About": self._description,
        #         },
        #     )


def main():
    # Запуск детектирования речевой активности в аудиовизуальном сигнале
    vad = RunVAD_GUI(lang="ru", path_to_logs="./openav/logs")
    vad.run(metadata=3, out=False)


if __name__ == "__main__":
    main()
