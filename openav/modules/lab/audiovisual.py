#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Мультимодальное объединение аудио- и видеомодальностей
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings

for warn in [
    UserWarning,
    FutureWarning,
]:
    warnings.filterwarnings(
        "ignore",
        category=warn,
    )

import os
import numpy as np

from dataclasses import dataclass  # Класс данных

# Типы данных
from typing import List, Dict

import datetime
import torch
from pathlib import Path  # Работа с путями в файловой системе
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from lion_pytorch import Lion
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Персональные
from openav.modules.core.exceptions import IsNestedCatalogsNotFoundError
from openav.modules.lab.audio import Audio  # Аудиомодальность
from openav.modules.lab.video import Video  # Видеомодальность
from openav.modules.nn.av_dataset import AVDataset, AVTest
from openav.modules.nn.utils import fix_seeds, train_one_epoch, val_one_epoch, save_conf_matrix
from openav.modules.nn.models import AVModel

# ######################################################################################################################
# Константы
# ######################################################################################################################

SUBFOLDERS: List[str] = ["train", "val", "test"]
SUBFOLDERS_TEST: List[str] = ["test"]
SHAPE_AUDIO: List[str] = ["channels", "n_mels", "samples"]
SHAPE_VIDEO: List[str] = ["frames", "channels", "width", "height"]
FIGSIZE_CONFUSION_MATRIX: List[str] = ["width", "height", "font_size", "dpi", "pad_inches"]
EXT_AV_VIDEO: List[str] = ["mov", "mp4", "webm"]  # Расширения искомых файлов
EXT_MODELS: str = "pt"
EXTH_MODELS: str = "pth"
OPTIMIZERS: List[str] = ["adam", "adamw", "sgd", "lion"]
REQUIRED_GRAD: List[str] = ["none", "a", "v", "av"]
CLASSES_TEST: List[str] = [
    "1_Позвонить",
    "2_Набрать_номер",
    "3_Отправить_сообщение",
    "6_Завершить_вызов",
    "7_Радио",
    "8_Музыка",
    "9_Воспроизвести",
    "11_Случа--и--ны--и--_выбор",
    "12_Отключить_случа--и--ны--и--_выбор",
    "13_Повтор",
    "16_Избегать_платных_дорог",
    "17_Карта",
    "19_Предыдущие_места_назначения",
    "20_Как_там_на_дорогах",
    "22_Надолго_пробка",
    "24_На_работу",
    "25_Максимальное_увеличение",
    "27_Остановить_маршрут",
    "28_Возобновить_маршрут",
    "29_Сколько_мне_еще_ехать",
    "30_Во_сколько_я_приеду",
    "32_Сбросить_маршрут",
    "34_На--и--ти_больницу",
    "36_На--и--ти_аптеку",
    "37_На--и--ти_банк",
    "39_На--и--ти_ресторан",
    "42_На--и--ти_железнодорожны--и--_вокзал",
    "47_Нет",
    "48_Предыдущая",
    "49_Следующая",
]


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class AVMessages(Audio, Video):
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

        self._subfolder_search: str = (
            self._('Поиск вложенной директории "{}" в директории "{}" (глубина вложенности: {})') + self._em
        )
        self._subfolders_search: str = (
            self._('Поиск вложенных директорий в директории "{}" (глубина вложенности: {})') + self._em
        )

        self._subfolder_not_found: str = self._("В указанной директории вложенная директория не найдена") + self._em

        self._files_audiovisual_find: str = (
            self._('Поиск файлов с расширениями "{}" в директории "{}" (глубина вложенности: {})') + self._em
        )

        self._files_audiovisual_find: str = (
            self._('Поиск файлов с расширениями "{}" в директории "{}" (глубина вложенности: {})') + self._em
        )

        self._sampling_nn: str = (
            self._("Разбиение найденных файлов на выборки (обучающая, валидационная, тестовая)") + self._em
        )

        self._sampling_nn_error: str = (
            self._(
                "Минимум одна выборка пустая (обучающая - {}, валидационная - {}, тестовая - {}) или количество меток "
                + "не совпадает с количеством файлов"
            )
            + self._em
        )

        self._sampling_nn_test_error: str = (
            self._("Тестовая выборка пустая или количество меток не совпадает с количеством файлов") + self._em
        )

        self._sampling_nn_true: str = self._("Обучающая - {} {}, валидационная - {} {}, тестовая - {} {}") + self._em
        self._format_percentage = lambda x: "({}%)".format(x)

        self._run_train: str = self._("Запуск процесса обучения") + self._em
        self._epoch: str = self._("Эпоха: {} из {}") + self._em
        self._loss: str = self._("Значения ошибки: обучение - {}, валидация - {}, тест - {}")
        self._acc_valid_and_test: str = self._("Валидация: точность {} | Тест: точность {}")
        self._acc_valid_up: str = (
            self._("Точность на валидационной выборке увеличилась ({} ---> {}). Сохранение модели") + self._em
        )
        self._acc_test: str = self._("Точность для тестовой выборке: {}")
        self._acc_test_up: str = self._("Точность на тестовой выборке увеличилась ({} ---> {})")

        self._end_train: str = self._("Процесс обучения завершен") + self._em

        self._run_test: str = self._("Запуск процесса тестирования") + self._em

        self._accuracy_score: str = self._("Точность на тестовой выборке: {}") + self._em
        self._create_confusion_matrix: str = self._("Создание матрицы спутывания") + self._em
        self._end_test: str = self._("Процесс тестирования завершен") + self._em


# ######################################################################################################################
# Мультимодальное
# ######################################################################################################################
@dataclass
class AV(AVMessages):
    """Класс для мультимодального объединения аудио- и видеомодальностей

    Args:
        path_to_logs (str): Смотреть :attr:`~openav.modules.core.logging.Logging.path_to_logs`
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        # ----------------------- Только для внутреннего использования внутри класса

        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.__model_session = "created_at_%d_%m_%Y_%H-%M"
        self.__cm_session = "confusion_matrix_%d_%m_%Y_%H-%M.png"

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    def __get_hierarchy_from_paths(self, paths, num_levels=0):
        hierarchies = []

        for path in paths:
            hierarchy = []
            while True:
                path, dir_name = os.path.split(path)
                if not dir_name:
                    hierarchy.append(path)
                    break
                hierarchy.append(dir_name)
            if num_levels > 0:
                hierarchy = hierarchy[:num_levels]
            if hierarchy[-1] in SUBFOLDERS:
                hierarchies.append(hierarchy)

        return hierarchies

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def train_audiovisual(
        self,
        subfolders: Dict[str, str],
        n_classes: int,
        classes: List[str],
        encoder_decoder: int,
        batch_size: int,
        max_segment: int,
        patience: int,
        epochs: int,
        seed: int,
        leaning_rate: float,
        weight_decay: float,
        optimizer: str,
        hidden_units: int,
        hidden_features: int,
        input_dim: int,
        shape_audio: Dict[str, int],
        shape_video: Dict[str, int],
        path_to_model_fa: str,
        path_to_model_fv: str,
        requires_grad: str,
        path_to_save_models: str,
        out: bool = True,
    ) -> bool:
        """Автоматическое обучение на аудиовизуальных данных

        Args:
            subfolders (Dict[str, str]): Словарь с подкаталогами с данными
            n_classes (int): Количество классов
            classes (List[str]): Список классов
            encoder_decoder (int): Количество энкодеров и декодеров
            batch_size (int): Размер батча
            max_segment (int): Максимальная длительность сегмента видео
            patience (int): Количество неудачных эпох
            epochs (int): Количество эпох
            seed (int): Начальное состояние обучения
            leaning_rate (float): Скорость обучения
            weight_decay (float): Скорость обучения
            optimizer (str): Оптимизатор
            hidden_units (int): Количество скрытых нейронов
            hidden_features (int): Количество скрытых признаков
            input_dim (int): Количество входных признаков
            shape_audio (Dict[str, int]): Входная размерность аудио лог-мел спектрограммы
            shape_video (Dict[str, int]): Входная размерность видеокадров
            path_to_model_fa (str): Путь к нейросетевой модели (аудио)
            path_to_model_fv (str): Путь к нейросетевой модели (видео)
            path_to_save_models (str): Путь к директории для сохранения моделей
            requires_grad (str): Заморозка слоев для извлечения ауди и видео признаков
            out (bool) Отображение

        Returns:
            bool: **True** если автоматическое обучение на аудиовизуальных данных произведено, в обратном случае
            **False**
        """

        try:
            # Проверка аргументов
            if (
                type(subfolders) is not dict
                or len(subfolders) == 0
                or not all(subfolder in subfolders for subfolder in SUBFOLDERS)
                or type(n_classes) is not int
                or not (0 < n_classes)
                or type(classes) is not list
                or len(classes) == 0
                or type(encoder_decoder) is not int
                or not (1 <= encoder_decoder <= 50)
                or type(batch_size) is not int
                or not 0 <= batch_size
                or type(max_segment) is not int
                or not (1 <= max_segment <= 10)
                or type(epochs) is not int
                or not (0 <= epochs <= 1000)
                or type(seed) is not int
                or not (0 < seed)
                or type(leaning_rate) is not float
                or type(weight_decay) is not float
                or type(hidden_units) is not int
                or type(optimizer) is not str
                or (optimizer in OPTIMIZERS) is False
                or not (0 < hidden_units)
                or type(hidden_features) is not int
                or not (0 < hidden_features)
                or type(patience) is not int
                or not (0 < patience)
                or type(input_dim) is not int
                or not (0 < input_dim)
                or type(shape_audio) is not dict
                or len(shape_audio) == 0
                or not all(shape in shape_audio for shape in SHAPE_AUDIO)
                or type(shape_video) is not dict
                or len(shape_video) == 0
                or not all(shape in shape_video for shape in SHAPE_VIDEO)
                or type(path_to_model_fa) is not str
                or not path_to_model_fa
                or type(path_to_model_fv) is not str
                or not path_to_model_fv
                or not Path(path_to_model_fa).is_file()
                or Path(path_to_model_fa).suffix.replace(".", "") != EXT_MODELS
                or not Path(path_to_model_fv).is_file()
                or Path(path_to_model_fv).suffix.replace(".", "") != EXT_MODELS
                or type(requires_grad) is not str
                or (requires_grad in REQUIRED_GRAD) is False
                or type(path_to_save_models) is not str
                or not path_to_save_models
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.train_audiovisual.__name__, out=out)
            return False
        else:
            depth = 3
            classes = [cls.lower() for cls in classes]
            optimizer = optimizer.lower()

            shape_audio = (
                None,
                max_segment,
                shape_audio[SHAPE_AUDIO[0]],
                shape_audio[SHAPE_AUDIO[1]],
                shape_audio[SHAPE_AUDIO[2]],
            )
            shape_video_frames = shape_video[SHAPE_VIDEO[0]]
            shape_video = (
                None,
                max_segment,
                shape_video[SHAPE_VIDEO[0]],
                shape_video[SHAPE_VIDEO[1]],
                shape_video[SHAPE_VIDEO[3]],
                shape_video[SHAPE_VIDEO[2]],
            )

            # Информационное сообщение
            self.message_info(
                self._subfolders_search.format(
                    self.message_line(self.path_to_dataset),
                    self.message_line(str(depth)),
                ),
                out=out,
            )

            nested_paths = self.get_paths(self.path_to_dataset, depth=depth, out=False)

            # Вложенные директории не найдены
            try:
                if len(nested_paths) == 0:
                    raise IsNestedCatalogsNotFoundError
            except IsNestedCatalogsNotFoundError:
                self.message_error(self._subfolders_not_found, space=self._space, out=out)
                return False

            # Информационное сообщение
            self.message_info(
                self._files_audiovisual_find.format(
                    self.message_line(", ".join(x.replace(".", "") for x in EXT_AV_VIDEO)),
                    self.message_line(self.path_to_dataset),
                    self.message_line(str(depth)),
                ),
                out=out,
            )

            hierarchy_from_paths = self.__get_hierarchy_from_paths(nested_paths, depth)

            # Информационное сообщение
            self.message_info(
                self._sampling_nn,
                out=out,
            )

            path_train, lb_train = [], []
            path_val, lb_val = [], []
            path_test, lb_test = [], []

            # Проход по всем вложенным директориям
            for nested_path in hierarchy_from_paths:
                nested_path_0 = nested_path[0].replace("--и--", "й").lower()
                if nested_path_0 in classes:
                    # Формирование списка с видеофайлами
                    for p in Path(os.path.join(self.path_to_dataset, *reversed(nested_path))).glob("*"):
                        # Добавление текущего пути к видеофайлу в список
                        if p.suffix.lower().replace(".", "") in EXT_AV_VIDEO:
                            [index for index, cls in enumerate(classes) if cls.lower() == nested_path_0]

                            if nested_path[-1] == subfolders[SUBFOLDERS[0]]:
                                path_train.append(p.resolve())
                                lb_train.append(classes.index(nested_path_0))
                            elif nested_path[-1] == subfolders[SUBFOLDERS[1]]:
                                if (nested_path[0] in CLASSES_TEST) is True:
                                    path_val.append(p.resolve())
                                    lb_val.append(classes.index(nested_path_0))
                            elif nested_path[-1] == subfolders[SUBFOLDERS[2]]:
                                if (nested_path[0] in CLASSES_TEST) is True:
                                    path_test.append(p.resolve())
                                    lb_test.append(classes.index(nested_path_0))
                            else:
                                pass

            # Директории с поднаборами данных не содержат визуальных файлов с необходимыми расширениями
            try:
                len_path_train = len(path_train)
                len_lb_train = len(lb_train)

                len_path_val = len(path_val)
                len_lb_val = len(lb_val)

                len_path_test = len(path_test)
                len_lb_test = len(lb_test)

                if (
                    len_path_train == 0
                    or len_lb_train == 0
                    or len_path_val == 0
                    or len_lb_val == 0
                    or len_path_test == 0
                    or len_lb_test == 0
                    or len_path_train != len_lb_train
                    or len_path_val != len_lb_val
                    or len_path_test != len_lb_test
                ):
                    raise ValueError
            except ValueError:
                self.message_error(
                    self._sampling_nn_error.format(
                        self.message_line(str(len_path_train)),
                        self.message_line(str(len_path_val)),
                        self.message_line(str(len_lb_test)),
                    ),
                    space=self._space,
                    out=out,
                )
                return False
            except Exception:
                self.message_error(self._unknown_err, space=self._space, out=out)
                return False
            else:
                total_samples = len_path_train + len_path_val + len_path_test

                train_percentage = round((len_path_train / total_samples) * 100, 2)
                val_percentage = round((len_path_val / total_samples) * 100, 2)
                test_percentage = round((len_path_test / total_samples) * 100, 2)

                self.message_info(
                    self._sampling_nn_true.format(
                        self.message_line(str(len_path_train)),
                        self.message_line(self._format_percentage(train_percentage)),
                        self.message_line(str(len_path_val)),
                        self.message_line(self._format_percentage(val_percentage)),
                        self.message_line(str(len_path_test)),
                        self.message_line(self._format_percentage(test_percentage)),
                    ),
                    space=self._space,
                    out=out,
                )

                train_data = AVDataset(
                    path_files=path_train,
                    labels=lb_train,
                    subset="train",
                    len_video=shape_video_frames,
                    max_segment=max_segment,
                )

                train_data = AVDataset(path_files=path_train, labels=lb_train, subset="train", max_segment=max_segment)
                test_data = AVDataset(path_files=path_test, labels=lb_test, subset="test", max_segment=max_segment)
                val_data = AVDataset(path_files=path_val, labels=lb_val, subset="val", max_segment=max_segment)

                train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                torch.autograd.set_detect_anomaly(True)

                stop_flag_training = 0

                fix_seeds(seed)

                model = AVModel(
                    shape_audio=shape_audio,
                    shape_video=shape_video,
                    input_dim=input_dim,
                    h_u=hidden_units,
                    h_f=hidden_features,
                    n_class=n_classes,
                    encoder_decoder=encoder_decoder,
                ).to(self.__device)
                model.feature_audio.load_state_dict(torch.load(path_to_model_fa))
                model.feature_video.load_state_dict(torch.load(path_to_model_fv))

                if requires_grad == REQUIRED_GRAD[0]:
                    for name, param in model.named_parameters():
                        param.requires_grad = True
                elif requires_grad == REQUIRED_GRAD[1]:
                    for name, param in model.named_parameters():
                        if any(layer_name.split(".")[0] in name for layer_name in ["feature_audio"]):
                            param.requires_grad = False
                elif requires_grad == REQUIRED_GRAD[2]:
                    for name, param in model.named_parameters():
                        if any(layer_name.split(".")[0] in name for layer_name in ["feature_video"]):
                            param.requires_grad = False
                elif requires_grad == REQUIRED_GRAD[3]:
                    for name, param in model.named_parameters():
                        if any(layer_name.split(".")[0] in name for layer_name in ["feature_audio", "feature_video"]):
                            param.requires_grad = False
                else:
                    for name, param in model.named_parameters():
                        param.requires_grad = True

                criterion = CrossEntropyLoss()

                if optimizer == OPTIMIZERS[0]:
                    optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate, weight_decay=weight_decay)
                elif optimizer == OPTIMIZERS[1]:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=leaning_rate, weight_decay=weight_decay)
                elif optimizer == OPTIMIZERS[2]:
                    optimizer = torch.optim.SGD(model.parameters(), lr=leaning_rate, weight_decay=weight_decay)
                elif optimizer == OPTIMIZERS[3]:
                    optimizer = Lion(model.parameters(), lr=leaning_rate, weight_decay=weight_decay)
                else:
                    pass

                max_acc = 0
                max_test_acc = 0

                date_time = datetime.datetime.now()
                date_path = date_time.strftime(self.__model_session)
                session_path = os.path.join(self.path_to_save_models, date_path)

                # Создание директории, где хранятся модели
                if self.create_folder(session_path, out=False) is False:
                    return False

                self.message_info(self._run_train, out=out)

                for e in range(1, epochs + 1, 1):
                    self.message_info(
                        self._epoch.format(
                            self.message_line(str(e)),
                            self.message_line(str(epochs)),
                        ),
                        out=out,
                    )

                    model.train()
                    avg_loss = train_one_epoch(train_dataloader, optimizer, criterion, model, self.__device)

                    model.eval()
                    acc, avg_vloss = val_one_epoch(val_dataloader, criterion, model, self.__device)

                    model.eval()
                    test_acc, avg_tloss = val_one_epoch(test_dataloader, criterion, model, self.__device)

                    round_f = 6

                    self.message_info(
                        self._loss.format(
                            self.message_line(str(round(avg_loss, round_f))),
                            self.message_line(str(round(avg_vloss, round_f))),
                            self.message_line(str(round(avg_tloss, round_f))),
                        ),
                        out=out,
                    )

                    if max_acc < acc:
                        stop_flag_training = 0

                        self.message_info(
                            self._acc_valid_and_test.format(
                                self.message_line(str(round(acc, round_f))),
                                self.message_line(str(round(test_acc, round_f))),
                            ),
                            out=out,
                        )
                        self.message_info(
                            self._acc_valid_up.format(
                                self.message_line(str(round(max_acc, round_f))),
                                self.message_line(str(round(acc, round_f))),
                            ),
                            out=out,
                        )
                        max_acc = acc
                        torch.save(model.state_dict(), "{}/e{}_{:.6f}.pth".format(session_path, e, acc))

                    if max_test_acc < test_acc:
                        self.message_info(
                            self._acc_test.format(
                                self.message_line(str(round(test_acc, round_f))),
                            ),
                            out=out,
                        )
                        self.message_info(
                            self._acc_test_up.format(
                                self.message_line(str(round(max_test_acc, round_f))),
                                self.message_line(str(round(test_acc, round_f))),
                            ),
                            out=out,
                        )

                        max_test_acc = test_acc

                    else:
                        stop_flag_training += 1

                    if stop_flag_training > patience:
                        self.message_info(self._end_train, out=out)

                        return True

    def test_audiovisual(
        self,
        subfolders: Dict[str, str],
        n_classes: int,
        classes: List[str],
        encoder_decoder: int,
        max_segment: int,
        hidden_units: int,
        hidden_features: int,
        input_dim: int,
        shape_audio: Dict[str, int],
        shape_video: Dict[str, int],
        save_confusion_matrix: bool,
        path_to_save_confusion_matrix: str,
        figsize_confusion_matrix: Dict[str, int],
        path_to_model: str,
        out: bool = True,
    ) -> bool:
        """Автоматическое тестирование на аудиовизуальных данных

        Args:
            subfolders (Dict[str, str]): Словарь с подкаталогами с данными
            n_classes (int): Количество классов
            classes (List[str]): Список классов
            encoder_decoder (int): Количество энкодеров и декодеров
            max_segment (int): Максимальная длительность сегмента видео
            hidden_units (int): Количество скрытых нейронов
            hidden_features (int): Количество скрытых признаков
            input_dim (int): Количество входных признаков
            shape_audio (Dict[str, int]): Входная размерность аудио лог-мел спектрограммы
            shape_video (Dict[str, int]): Входная размерность видеокадров
            save_confusion_matrix (bool): Сохранение матрицы спутывания
            path_to_save_confusion_matrix (str):  Путь к директории для сохранения матрицы спутывания
            figsize_confusion_matrix (Dict[str, int]): Настройки для формирования изображения матрицы спутывания
            path_to_model (str): Путь к нейросетевой аудиовизуальной модели
            out (bool) Отображение

        Returns:
            bool: **True** если автоматическое тестирование на аудиовизуальных данных произведено, в обратном случае
            **False**
        """

        try:
            # Проверка аргументов
            if (
                type(subfolders) is not dict
                or len(subfolders) == 0
                or not all(subfolder in subfolders for subfolder in SUBFOLDERS_TEST)
                or type(n_classes) is not int
                or not (0 < n_classes)
                or type(classes) is not list
                or len(classes) == 0
                or type(encoder_decoder) is not int
                or not (1 <= encoder_decoder <= 50)
                or type(max_segment) is not int
                or not (1 <= max_segment <= 10)
                or not (0 < hidden_units)
                or type(hidden_features) is not int
                or not (0 < hidden_features)
                or type(input_dim) is not int
                or not (0 < input_dim)
                or type(shape_audio) is not dict
                or len(shape_audio) == 0
                or not all(shape in shape_audio for shape in SHAPE_AUDIO)
                or type(shape_video) is not dict
                or len(shape_video) == 0
                or not all(shape in shape_video for shape in SHAPE_VIDEO)
                or type(save_confusion_matrix) is not bool
                or type(path_to_save_confusion_matrix) is not str
                or not path_to_save_confusion_matrix
                or len(figsize_confusion_matrix) == 0
                or not all(k in figsize_confusion_matrix for k in FIGSIZE_CONFUSION_MATRIX)
                or type(path_to_model) is not str
                or not path_to_model
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.test_audiovisual.__name__, out=out)
            return False
        else:
            depth = 3
            classes = [cls.lower() for cls in classes]

            shape_audio = (
                None,
                max_segment,
                shape_audio[SHAPE_AUDIO[0]],
                shape_audio[SHAPE_AUDIO[1]],
                shape_audio[SHAPE_AUDIO[2]],
            )
            shape_video = (
                None,
                max_segment,
                shape_video[SHAPE_VIDEO[0]],
                shape_video[SHAPE_VIDEO[1]],
                shape_video[SHAPE_VIDEO[3]],
                shape_video[SHAPE_VIDEO[2]],
            )

            # Информационное сообщение
            self.message_info(
                self._subfolder_search.format(
                    self.message_line(subfolders[SUBFOLDERS_TEST[0]]),
                    self.message_line(self.path_to_dataset),
                    self.message_line(str(depth)),
                ),
                out=out,
            )

            nested_paths = self.get_paths(self.path_to_dataset, depth=depth, out=False)

            # Вложенные директории не найдены
            try:
                if len(nested_paths) == 0:
                    raise IsNestedCatalogsNotFoundError
            except IsNestedCatalogsNotFoundError:
                self.message_error(self._subfolder_not_found, space=self._space, out=out)
                return False

            # Информационное сообщение
            self.message_info(
                self._files_audiovisual_find.format(
                    self.message_line(", ".join(x.replace(".", "") for x in EXT_AV_VIDEO)),
                    self.message_line(self.path_to_dataset),
                    self.message_line(str(depth)),
                ),
                out=out,
            )

            hierarchy_from_paths = self.__get_hierarchy_from_paths(nested_paths, depth)

            path_test, lb_test = [], []

            # Проход по всем вложенным директориям
            for nested_path in hierarchy_from_paths:
                nested_path_0 = nested_path[0].replace("--и--", "й").lower()
                if nested_path_0 in classes:
                    # Формирование списка с видеофайлами
                    for p in Path(os.path.join(self.path_to_dataset, *reversed(nested_path))).glob("*"):
                        # Добавление текущего пути к видеофайлу в список
                        if p.suffix.lower().replace(".", "") in EXT_AV_VIDEO:
                            [index for index, cls in enumerate(classes) if cls.lower() == nested_path[0].lower()]

                            if nested_path[-1] == subfolders[SUBFOLDERS_TEST[0]]:
                                path_test.append(p.resolve())
                                lb_test.append(classes.index(nested_path_0))
                            else:
                                pass

            lb_test_names = [classes[val] for val in set(lb_test)]

            # Директории с поднаборами данных не содержат визуальных файлов с необходимыми расширениями
            try:
                len_path_test = len(path_test)
                len_lb_test = len(lb_test)

                if len_path_test == 0 or len_lb_test == 0 or len_path_test != len_lb_test:
                    raise ValueError
            except ValueError:
                self.message_error(
                    self._sampling_nn_test_error,
                    space=self._space,
                    out=out,
                )
                return False
            except Exception:
                self.message_error(self._unknown_err, space=self._space, out=out)
                return False
            else:
                model = AVModel(
                    shape_audio=shape_audio,
                    shape_video=shape_video,
                    input_dim=input_dim,
                    h_u=hidden_units,
                    h_f=hidden_features,
                    n_class=n_classes,
                    encoder_decoder=encoder_decoder,
                ).to(self.__device)

                model.load_state_dict(torch.load(path_to_model, map_location=torch.device(self.__device)))

                self.message_info(self._run_test, out=out)

                processor = AVTest(max_segment=max_segment)

                torch.autograd.set_detect_anomaly(True)

                model.eval().to(self.__device)

                preds = []

                for path in tqdm(path_test):
                    _, audio_data, video_data = processor.get_metadata(name_file=str(path))
                    with torch.no_grad():
                        prob = model(audio_data.to(self.__device), video_data.to(self.__device)).cpu().detach().numpy()
                    pred = np.argmax(prob)
                    preds.append(pred)

                self.message_info(
                    self._accuracy_score.format(
                        self.message_line(str(round(accuracy_score(lb_test, preds) * 100, 2)) + "%"),
                    ),
                    out=out,
                )

                if save_confusion_matrix:
                    self.message_info(self._create_confusion_matrix, out=out)

                    date_time = datetime.datetime.now()
                    date_path = date_time.strftime(self.__cm_session)

                    save_conf_matrix(
                        y_true=lb_test,
                        y_pred=preds,
                        name_labels=["_".join(val.split("_")[1:]).capitalize() for val in lb_test_names],
                        filename=os.path.join(path_to_save_confusion_matrix, date_path),
                        figsize_w=figsize_confusion_matrix[FIGSIZE_CONFUSION_MATRIX[0]],
                        figsize_h=figsize_confusion_matrix[FIGSIZE_CONFUSION_MATRIX[1]],
                        font_size=figsize_confusion_matrix[FIGSIZE_CONFUSION_MATRIX[2]],
                        dpi=figsize_confusion_matrix[FIGSIZE_CONFUSION_MATRIX[3]],
                        pad_inches=figsize_confusion_matrix[FIGSIZE_CONFUSION_MATRIX[4]],
                    )

                self.message_info(self._end_test, out=out)

                return True
