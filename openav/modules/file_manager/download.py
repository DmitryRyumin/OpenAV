#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Загрузка файлов
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
import numpy as np  # Научные вычисления
import requests  # Отправка HTTP запросов
import re  # Регулярные выражения
import shutil  # Набор функций высокого уровня для обработки файлов, групп файлов, и папок

from pathlib import Path  # Работа с путями в файловой системе

# Персональные
from openav.modules.core.exceptions import InvalidContentLength
from openav.modules.file_manager.file_manager import FileManager  # Работа с файлами


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class DownloadMessages(FileManager):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._could_not_process_url = self._("Не удалось обработать указанный URL") + self._em
        self._url_incorrect = self._("URL указан некорректно") + self._em
        self._url_incorrect_content_length = self._("Не определен размер файла для загрузки") + self._em
        self._automatic_download: str = self._("Загрузка файла") + ' "{}"' + self._em
        self._url_error_code_http: str = "( " + self._("ошибка") + "{})"
        self._url_error_http: str = self._("не удалось скачать файл") + ' "{}"{}' + self._em


# ######################################################################################################################
# Загрузка файлов
# ######################################################################################################################
class Download(DownloadMessages):
    """Класс для загрузки файлов"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._headers: str = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/89.0.4389.90 Safari/537.36"
        )  # User-Agent

        self._url_last_filename: str = ""  # Имя последнего загруженного файла

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    def __progressbar_download_file_from_url(
        self, url_filename: str, progress: float, clear_out: bool = True, last: bool = False, out: bool = True
    ) -> None:
        """Индикатор выполнения загрузки файла из URL

        .. note::
            private (приватный метод)
        """

        if clear_out is False and last is True:
            clear_out, last = last, clear_out
        elif clear_out is False and last is False:
            clear_out = True

        try:
            # Проверка аргументов
            if (
                type(url_filename) is not str
                or not url_filename
                or type(progress) is not float
                or not (0 <= progress <= 100)
            ):
                raise TypeError
        except TypeError:
            self._inv_args(__class__.__name__, self.__progressbar_download_file_from_url.__name__, out=out)
            return None

        self._info(
            self._automatic_download.format(self._info_wrapper(url_filename)) + self._download_precent.format(progress),
            last=last,
            out=False,
        )

        if out:
            self.show_notebook_history_output()  # Отображение истории вывода сообщений в ячейке Jupyter

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def download_file_from_url(self, url: str, force_reload: bool = True, out: bool = True) -> int:
        """Загрузка файла из URL

        Args:
            url (str): Полный путь к файлу
            force_reload (bool): Принудительная загрузка файла из сети
            out (bool): Отображение

        Returns:
            int: Код статуса ответа:

                * ``200`` - Файл загружен
                * ``400`` - Ошибка при проверке аргументов
                * ``404`` - Не удалось скачать файл
        """

        try:
            # Проверка аргументов
            if type(url) is not str or not url or type(force_reload) is not bool or type(out) is not bool:
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.download_file_from_url.__name__, out=out)
            return 400
        else:
            try:
                # Отправка GET запроса для получения файла
                r = requests.get(url, headers={"user-agent": self._headers}, stream=True)
            except (
                # https://requests.readthedocs.io/en/master/_modules/requests/exceptions/
                requests.exceptions.MissingSchema,
                requests.exceptions.InvalidSchema,
                requests.exceptions.ConnectionError,
                requests.exceptions.InvalidURL,
            ):
                self.message_error(self._could_not_process_url, out=out)
                return 404
            except Exception:
                self.message_error(self._unknown_err, out=out)
                return 404
            else:
                # Имя файла
                if "Content-Disposition" in r.headers.keys():
                    try:
                        url_filename = re.findall(r'(?<=[\(\{\["]).+(?=[\)\}\]"])', r.headers["Content-Disposition"])[0]
                    except IndexError:
                        url_filename = re.findall(
                            r'filename\*?=[\'"]?(?:UTF-\d[\'"]*)?([^;\r\n"\']*)[\'"]?;?',
                            r.headers["Content-Disposition"],
                        )[0]
                else:
                    url_filename = url.split("/")[-1]

                try:
                    # URL файл невалидный
                    if not url_filename or not Path(url_filename).suffix:
                        if not Path(url_filename).stem.lower():
                            raise requests.exceptions.InvalidURL

                        if r.headers["Content-Type"] == "image/jpeg":
                            ext = "jpg"
                        elif r.headers["Content-Type"] == "image/png":
                            ext = "png"
                        elif r.headers["Content-Type"] == "text/plain":
                            ext = "txt"
                        elif r.headers["Content-Type"] == "text/csv":
                            ext = "csv"
                        elif r.headers["Content-Type"] == "video/mp4":
                            ext = "mp4"
                        else:
                            raise requests.exceptions.InvalidHeader

                        url_filename = Path(url_filename).stem + "." + ext
                except (requests.exceptions.InvalidURL, requests.exceptions.InvalidHeader):
                    self.message_error(self._url_incorrect, out=out)
                    return 404
                except Exception:
                    self.message_error(self._unknown_err, out=out)
                    return 404
                else:
                    # Информационное сообщение
                    self.message_info(self._automatic_download.format(self.message_line(url_filename)), out=out)

                    # Создание директории для сохранения файла
                    if self.create_folder(self.path_to_save_models, out=False) is False:
                        return 404

                    local_file = os.path.join(self.path_to_save_models, url_filename)  # Путь к файлу

                    try:
                        # Принудительная загрузка файла из сети
                        if force_reload is True:
                            # Файл найден
                            if os.path.isfile(local_file) is True:
                                # Удаление файла
                                try:
                                    shutil.rmtree(local_file)
                                except OSError:
                                    os.remove(local_file)
                                except Exception:
                                    raise Exception
                    except Exception:
                        self.message_error(self._unknown_err, space=self._space, out=out)
                        return 404
                    else:
                        # Файл с указанным именем найден локально и принудительная загрузка файла из сети не указана
                        if Path(local_file).is_file() is True and force_reload is False:
                            self._url_last_filename = local_file
                            return 200
                        else:
                            # Ответ получен
                            if r.status_code == 200:
                                total_length = int(r.headers.get("content-length", 0))  # Длина файла

                                try:
                                    if total_length == 0:
                                        raise InvalidContentLength
                                except InvalidContentLength:
                                    self.message_error(self._url_incorrect_content_length, space=self._space, out=out)
                                    return 404
                                else:
                                    num_bars = int(np.ceil(total_length / self.chunk_size))  # Количество загрузок

                                    try:
                                        # Открытие файла для записи
                                        with open(local_file, "wb") as f:
                                            # Индикатор выполнения
                                            self.__progressbar_download_file_from_url(
                                                url_filename, 0.0, clear_out=True, last=True, out=out
                                            )

                                            # Сохранение файла по частям
                                            for i, chunk in enumerate(r.iter_content(chunk_size=self.chunk_size_)):
                                                f.write(chunk)  # Запись в файл
                                                f.flush()

                                                # Индикатор выполнения
                                                self.__progressbar_download_file_from_url(
                                                    url_filename,
                                                    round(i * 100 / num_bars, 2),
                                                    clear_out=True,
                                                    last=True,
                                                    out=out,
                                                )

                                            # Индикатор выполнения
                                            self.__progressbar_download_file_from_url(
                                                url_filename, 100.0, clear_out=True, last=True, out=out
                                            )
                                    except Exception:
                                        self._other_error(self._unknown_err, out=out)
                                        return 404
                                    else:
                                        self._url_last_filename = local_file
                                        return 200
                            else:
                                self._error(
                                    self._url_error_http.format(
                                        self._info_wrapper(url_filename),
                                        self._url_error_code_http.format(self._error_wrapper(str(r.status_code))),
                                    ),
                                    out=out,
                                )

                                return 404
