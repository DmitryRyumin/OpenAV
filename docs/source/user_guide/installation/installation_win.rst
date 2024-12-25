.. include:: ../../reuse_content/general.rst

.. |a_download| raw:: html

    <a href="https://www.python.org/downloads/windows/" target="_blank">

Установка и обновление на операционной системе Windows
======================================================

Подготовка рабочего пространства и оздание изолированной среды OpenAV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Перейти на сайт и |a_download| скачать необходимый дистрибутив |/a| (например Python 3.12.8)

.. figure:: ../../_static/img/installation/win/download.png
        :align: center
        :alt: Загрузка дистрибутива

        |br|

2. Установить Python

.. figure:: ../../_static/img/installation/win/install.png
        :align: center
        :alt: Завершенный процесс установки Python

        |br|

3. Перейти в подготовленный каталог с проектом:

.. code-block:: sh

   cd ПУТЬ_К_ДИРЕКТОРИИ

.. figure:: ../../_static/img/installation/win/dir.png
        :align: center
        :alt: Переход в директорию с проектом через команду «cd»

        |br|

4. Разместить новую виртуальную среду OpenAV в указанной директории с указанной версией Python (с помощью команды ``py --list`` можно проверить установленные версии Python):

.. code-block:: sh

   py -3.12 -m venv env

.. figure:: ../../_static/img/installation/win/env.png
        :align: center
        :alt: Размещение новой виртуальной среды OpenAV в указанной директории с указанной версией Python

        |br|

5. Активировать созданную виртуальную среду OpenAV в директории:

.. code-block:: sh

   cd env/Scripts/
   ./activate
   cd ../../

.. figure:: ../../_static/img/installation/win/activate.png
        :align: center
        :alt: Активация созданной виртуальной среды OpenAV в директории с проектом

        |br|

Работа с библиотекой OpenAV
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Установить библиотеку OpenAV:

.. code-block:: sh

   pip install openav

.. figure:: ../../_static/img/installation/win/openav.png
        :align: center
        :alt: Установленная библиотека OpenAV со всеми зависимостями

        |br|
