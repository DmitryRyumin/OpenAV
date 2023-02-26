#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Работа с файлами
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings

for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

import os  # Работа с файловой системой
import re  # Регулярные выражения
import shutil  # Набор функций высокого уровня для обработки файлов, групп файлов, и папок
from pathlib import Path  # Работа с путями в файловой системе

from dataclasses import dataclass  # Класс данных

from typing import List, Optional, Iterable  # Типы данных

# Персональные
from openav.modules.core.core import Core  # Ядро


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class FileManagerMessages(Core):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._folder_not_found: str = self._('Директория "{}" не найдена') + self._em
        self._file_name: str = self._("Необходимо указать название файла с расширением") + " {}" + self._em
        self._dir_found: str = self._("Вместо файла передана директория") + self._em
        self._file_find_hide: str = self._('Поиск "{}" файла') + self._em
        self._file_find: str = self._file_find_hide
        self._wrong_extension: str = self._("Расширение файла должно быть") + " {}" + self._em
        self._file_not_found_create: str = self._('Файл "{}" не найден, но был создан') + self._em
        self._file_not_found: str = self._('Файл "{}" не найден') + self._em
        self._dir_name: str = self._("Необходимо указать название директории") + self._em
        self._files_find: str = self._('Поиск файлов с расширениями "{}" в директории "{}"') + self._em
        self._files_not_found: str = self._("В указанной директории необходимые файлы не найдены") + self._em
        self._create_folder: str = self._('Создание директории "{}"') + self._em
        self._folder_not_create: str = self._('Директория "{}" не создана') + self._em
        self._clear_folder: str = self._('Очистка директории "{}"') + self._em
        self._clear_folder_not_found: str = self._('Директория "{}" не найдена') + self._em


# ######################################################################################################################
# Работа с файлами
# ######################################################################################################################
@dataclass
class FileManager(FileManagerMessages):
    """Класс для работы с файлами"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

    # ------------------------------------------------------------------------------------------------------------------
    #  Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def re_inv_chars(path: str) -> str:
        """Удаление недопустимых символов из пути

        Args:
            path (str): Путь

        Returns:
            str: Путь
        """

        return re.sub('[\\/:"*?<>|]+', "", path)

    def get_paths(self, path: Iterable, depth: int = 1, out: bool = True) -> List[Optional[str]]:
        """Получение поддиректорий

        Args:
            path (Iterable): Путь к директории
            depth (int): Глубина иерархии для извлечения поддиректорий
            out (bool): Отображение

        Returns:
            List[Optional[str]]: Список с поддиректориями
        """

        try:
            # Проверка аргументов
            if (
                not isinstance(path, Iterable)
                or not path
                or type(depth) is not int
                or depth < 1
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.get_paths.__name__, out=out)
            return []
        else:
            if type(path) is not list:
                path = [path]

            new_path = []  # Список с директориями

            # Проход по всем директориям
            for curr_path in path:
                try:
                    scandir = os.scandir(os.path.normpath(str(curr_path)))
                except FileNotFoundError:
                    self.message_error(self._folder_not_found.format(self.message_line(str(curr_path))), out=out)
                    return []
                except Exception as e:
                    self.message_error(self._unknown_err, out=out)
                    print(e)
                    return []
                else:
                    for f in scandir:
                        if f.is_dir() and not f.name.startswith("."):
                            ignore = False  # По умолчанию не игнорировать директорию
                            if depth == 1:
                                for curr_dir in self.ignore_dirs:
                                    if type(curr_dir) is not str:
                                        continue
                                    # Игнорировать директорию
                                    if re.search("^" + curr_dir, f.name) is not None:
                                        ignore = True

                            if ignore is False:
                                new_path.append(f.path)

            # Рекурсивный переход на следующий уровень иерархии
            if depth != 1 and len(new_path) > 0:
                return self.get_paths(new_path, depth - 1)

            return new_path  # Список с директориями

    def search_file(self, path_to_file: str, ext: str, create: bool = False, out: bool = True) -> bool:
        """Поиск файла

        Args:
            path_to_file (str): Путь к файлу
            ext (str): Расширение файла
            create (bool): Создание файла в случае его отсутствия
            out (bool): Печатать процесс выполнения

        Returns:
            bool: **True** если файл найден, в обратном случае **False**
        """

        # Проверка аргументов
        if (
            type(path_to_file) is not str
            or type(ext) is not str
            or not ext
            or type(create) is not bool
            or type(out) is not bool
        ):
            self.inv_args(__class__.__name__, self.search_file.__name__, out=out)
            return False

        # Файл не передан
        if not path_to_file:
            self.message_error(self._file_name.format(ext.lower()), out=out)
            return False

        path_to_file = os.path.normpath(path_to_file)
        ext = ext.replace(".", "")

        # Передана директория
        if os.path.isdir(path_to_file) is True:
            self.message_error(self._dir_found, out=out)
            return False

        # Вывод сообщения
        self.message_info(self._file_find.format(self.message_line(os.path.basename(path_to_file))), out=out)

        self._file_load = self._file_find_hide  # Установка сообщения в исходное состояние

        _, extension = os.path.splitext(path_to_file)  # Расширение файла

        if ext != extension.replace(".", ""):
            self.message_error(self._wrong_extension.format(ext), space=self._space, out=out)
            return False

        # Файл не найден
        if os.path.isfile(path_to_file) is False:
            # Создание файла
            if create is True:
                open(path_to_file, "a", encoding="utf-8").close()

                self.message_info(
                    self._file_not_found_create.format(os.path.basename(path_to_file)), space=self._space, out=out
                )
                return False

            self.message_error(self._file_not_found.format(os.path.basename(path_to_file)), space=self._space, out=out)
            return False

        return True  # Результат

    def search_files(
        self, path_to_folder: str, exts: List[str], sort: bool = True, out: bool = True
    ) -> List[Optional[str]]:
        """Поиск файлов в указанной директории

        Args:
            path_to_folder (str): Путь к директории с файлами
            exts (List[str]): Расширения файлов
            sort (bool): Сортировать файлы
            out (bool): Печатать процесс выполнения

        Returns:
            List[Optional[str]]: список с найденными файлами
        """

        # Проверка аргументов
        if (
            type(path_to_folder) is not str
            or type(exts) is not list
            or len(exts) == 0
            or type(sort) is not bool
            or type(out) is not bool
        ):
            self.inv_args(__class__.__name__, self.search_files.__name__, out=out)
            return []

        path_to_folder = os.path.normpath(path_to_folder)
        exts = [ext.replace(".", "") for ext in exts]

        # Директория не передана
        if os.path.isdir(path_to_folder) is False:
            self.message_error(self._dir_name, out=out)
            return []

        # Вывод сообщения
        self.message_info(self._files_find.format(", ".join(x for x in exts), path_to_folder), out=out)

        # Список из файлов с необходимым расширением
        files = [str(p.resolve()) for p in Path(path_to_folder).glob("*") if p.suffix.replace(".", "") in exts]

        # В указанной директории не найдены необходимые файлы
        if len(files) == 0:
            self.message_error(self._files_not_found, space=self._space, out=out)
            return []

        # Сортировка файлов
        if sort is True:
            return sorted(files)

        return files

    def create_folder(self, path_to_folder: str, out: bool = True) -> bool:
        """Создание директории

        Args:
            path_to_folder (str): Путь к директории
            out (bool): Печатать процесс выполнения

        Returns:
            bool: **True** если директория создана, в обратном случае **False**
        """

        # Проверка аргументов
        if type(path_to_folder) is not str or not path_to_folder or type(out) is not bool:
            self.inv_args(__class__.__name__, self.create_folder.__name__, out=out)
            return False

        path_to_folder = os.path.normpath(path_to_folder)

        if not os.path.exists(path_to_folder):
            self.message_info(self._create_folder.format(path_to_folder), out=out)

            try:
                os.makedirs(path_to_folder)
            except Exception:
                self.message_error(self._folder_not_create.format(path_to_folder), space=4, out=out)
                return False
            else:
                return True
        else:
            return True

    def clear_folder(self, path_to_folder: str, out: bool = True) -> bool:
        """Очистка директории

        Args:
            path_to_folder (str): Путь к директории
            out (bool): Печатать процесс выполнения

        Returns:
            bool: **True** если директория очищена, в обратном случае **False**
        """

        # Проверка аргументов
        if type(path_to_folder) is not str or not path_to_folder or type(out) is not bool:
            self.inv_args(__class__.__name__, self.clear_folder.__name__, out=out)
            return False

        path_to_folder = os.path.normpath(path_to_folder)

        # Вывод сообщения
        self.message_info(self._clear_folder.format(path_to_folder), out=out)

        # Каталог с файлами найден
        if os.path.exists(path_to_folder):
            # Очистка
            for filename in os.listdir(path_to_folder):
                filepath = os.path.join(path_to_folder, filename)
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)
        else:
            self.message_error(self._clear_folder_not_found.format(path_to_folder), out=out)
            return False

        return True  # Результат
