#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Обработка архивов
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings

for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

from dataclasses import dataclass  # Класс данных

import os  # Взаимодействие с файловой системой
from zipfile import ZipFile, BadZipFile  # Работа с ZIP архивами
from pathlib import Path  # Работа с путями в файловой системе
import shutil  # Набор функций высокого уровня для обработки файлов, групп файлов, и папок

from typing import List, Optional  # Типы данных

# Персональные
from openav.modules.core.core import Core  # Ядро

# ######################################################################################################################
# Константы
# ######################################################################################################################
EXTS_ZIP: List[str] = ["zip"]  # Поддерживаемые расширения архивов


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class UnzipMessages(Core):
    """Класс для сообщений

    Args:
        path_to_logs (str): Смотреть :attr:`~openav.modules.core.logging.Logging.path_to_logs`
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._automatic_unzip: str = self._('Разархивирование архива "{}"') + self._em
        self._download_precent: str = " {}%" + self._em
        self._automatic_unzip_progress: str = self._automatic_unzip + " {}%" + self._em
        self._error_unzip: str = self._('Не удалось разархивировать архив "{}"') + self._em
        self._error_rename: str = self._('Не удалось переименовать директорию из "{}" в "{}"') + self._em


# ######################################################################################################################
# Обработка архивов
# ######################################################################################################################
class Unzip(UnzipMessages):
    """Класс для обработки архивов

    Args:
        path_to_logs (str): Смотреть :attr:`~openav.modules.core.logging.Logging.path_to_logs`
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._path_to_unzip: str = ""  # Имя директории для разархивирования

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def path_to_unzip(self) -> str:
        """Получение директории для разархивирования

        Returns:
            str: Директория для разархивирования
        """

        return self._path_to_unzip

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------------------------------------------------------

    # Индикатор выполнения
    def __progressbar_unzip(self, path_to_zipfile: str, item: float, out: bool):
        """
        Индикатор выполнения

        Аргументы:
            path_to_zipfile - Путь до архива
            item - Процент выполнения
            out - Отображение
        """

        self._info(
            self._automatic_unzip.format(self._info_wrapper(Path(path_to_zipfile).name))
            + self._download_precent.format(item),
            last=True,
            out=False,
        )

        if out:
            self.show_notebook_history_output()

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def unzip(
        self, path_to_zipfile: str, new_name: Optional[str] = None, force_reload: bool = True, out: bool = True
    ) -> bool:
        """Разархивирование архива

        Args:
            path_to_zipfile (str): Полный путь до архива
            new_name (str): Имя директории для разархивирования
            force_reload (bool): Принудительное разархивирование
            out (bool): Отображение

        Returns:
            bool: **True** если разархивирование прошло успешно, в обратном случае **False**
        """

        try:
            if new_name is None:
                new_name = path_to_zipfile  # Имя директории для разархивирования не задана

            # Проверка аргументов
            if (
                type(path_to_zipfile) is not str
                or not path_to_zipfile
                or type(new_name) is not str
                or not new_name
                or type(force_reload) is not bool
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.unzip.__name__, out=out)
            return False
        else:
            # Нормализация путей
            path_to_zipfile = os.path.normpath(path_to_zipfile)
            new_name = os.path.normpath(new_name)

            # Информационное сообщение
            self.message_info(
                self._automatic_unzip.format(self.message_line(Path(path_to_zipfile).name)), end=False, out=out
            )

            # Имя директории для разархивирования
            if path_to_zipfile == new_name:
                self._path_to_unzip = str(Path(path_to_zipfile).with_suffix(""))
            else:
                self._path_to_unzip = os.path.join(self.path_to_save_models, Path(new_name).name)

            try:
                # Расширение файла неверное
                if (Path(path_to_zipfile).suffix.replace(".", "") in EXTS_ZIP) is False:
                    raise TypeError
            except TypeError:
                self.message_error(
                    self._wrong_extension.format(self.message_line(", ".join(x for x in EXTS_ZIP))),
                    space=self._space,
                    start=True,
                    out=out,
                )
                return False
            else:
                # Принудительное разархивирование отключено
                if force_reload is False:
                    # Каталог уже существует
                    if os.path.isdir(self._path_to_unzip):
                        return True
                try:
                    # Файл не найден
                    if os.path.isfile(path_to_zipfile) is False:
                        raise FileNotFoundError
                except FileNotFoundError:
                    self.message_error(
                        self._file_not_found.format(self.message_line(Path(path_to_zipfile).name)),
                        space=self._space,
                        start=True,
                        out=out,
                    )
                    return False
                except Exception:
                    self.message_error(self._unknown_err, space=self._space, start=True, out=out)
                    return False
                else:
                    extracted_size = 0  # Объем извлеченной информации

                    try:
                        # Процесс разархивирования
                        with ZipFile(path_to_zipfile, "r") as zf:
                            uncompress_size = sum((file.file_size for file in zf.infolist()))  # Общий размер
                            # Проход по всем файлам, которые необходимо разархивировать
                            for file in zf.infolist():
                                extracted_size += file.file_size  # Увеличение общего объема
                                zf.extract(file, self.path_to_save_models)  # Извлечение файла из архива

                                # Индикатор выполнения
                                self.message_progressbar(
                                    self._automatic_unzip_progress.format(
                                        self.message_line(path_to_zipfile),
                                        round(extracted_size * 100 / uncompress_size, 2),
                                    ),
                                    out=out,
                                )

                            # Индикатор выполнения
                            self.message_progressbar(
                                self._automatic_unzip_progress.format(self.message_line(path_to_zipfile), 100),
                                close=True,
                                out=out,
                            )
                    except BadZipFile:
                        self.message_error(
                            self._error_unzip.format(self.message_line(Path(path_to_zipfile).name)), out=out
                        )
                        return False
                    except Exception:
                        self.message_error(self._unknown_err, out=out)
                        return False
                    else:
                        # Переименовывать директорию не нужно
                        if path_to_zipfile == new_name:
                            return True

                        try:
                            # Принудительное разархивирование включено и каталог уже существует
                            if force_reload is True and os.path.isdir(self._path_to_unzip):
                                # Удаление директории
                                try:
                                    shutil.rmtree(self._path_to_unzip)
                                except OSError:
                                    os.remove(self._path_to_unzip)
                                except Exception:
                                    raise Exception
                        except Exception:
                            self.message_error(self._unknown_err, out=out)
                            return False
                        else:
                            try:
                                # Переименование
                                os.rename(Path(path_to_zipfile).with_suffix(""), self._path_to_unzip)
                            except Exception:
                                self.message_error(
                                    self._error_rename.format(
                                        self.message_line(Path(path_to_zipfile).with_suffix("")),
                                        self.message_line(Path(new_name).name),
                                    ),
                                    out=out,
                                )
                                return False
                            else:
                                return True
