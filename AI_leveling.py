## Загрузка библиотек


# Работа с массивами
import numpy as np

# Генератор аугментированных изображений
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Основа для создания последовательной модели
from tensorflow.keras.models import Sequential

# Основные слои
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

# Оптимизатор
from tensorflow.keras.optimizers import Adam

# Матрица ошибок классификатора
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Подключение модуля для загрузки данных из облака
import gdown

# Инструменты для работы с файлами
import os

# Отрисовка графиков
import matplotlib.pyplot as plt

# Рисование графиков в ячейках Colab
%matplotlib inline

# Отрисовка изображений
from PIL import Image

# Генерация случайных чисел
import random

## Задание гиперпараметров модели


# Задание гиперпараметров

TRAIN_PATH          = '/content/pict/'       # Папка для обучающего набора данных
TEST_PATH           = '/content/pict_test/'  # Папка для тестового набора данных

TEST_SPLIT          = 0.1                   # Доля тестовых данных в общем наборе
VAL_SPLIT           = 0.2                   # Доля проверочной выборки в обучающем наборе

# Задание единых размеров изображений

IMG_LEFT = 0
IMG_TOP =  180
IMG_RIGHT = 168
IMG_BOTTOM = 348
IMG_HEIGHT = 84          # Высота изображения
IMG_WIDTH = 84           # Ширина изображения
IMG_CHANNELS = 3         # Количество каналов (для RGB равно 3, для Grey равно 1)

# Параметры аугментации
ROTATION_RANGE      = 0                     # Пределы поворота
WIDTH_SHIFT_RANGE   = 0.15                  # Пределы сдвига по горизонтали
HEIGHT_SHIFT_RANGE  = 0.00                  # Пределы сдвига по вертикали
ZOOM_RANGE          = 0.15                  # Пределы увеличения/уменьшения
BRIGHTNESS_RANGE    = (0.7, 1.3)            # Пределы изменения яркости
HORIZONTAL_FLIP     = True                  # Горизонтальное отражение разрешено

EPOCHS              = 60                    # Число эпох обучения
BATCH_SIZE          = 24                    # Размер батча для обучения модели
OPTIMIZER           = Adam(0.00001)          # Оптимизатор

##Загрузка датасета и подготовка данных

# Подключение модуля работы с диском
from google.colab import drive

# Авторизация необходима
drive.mount('/content/drive')

# Очистка данных от прошлого запуска (если есть)
!rm -rf {TRAIN_PATH} {TEST_PATH}

# Разархивация датасета в директорию данных
!unzip -qo "/content/drive/My Drive/dataset/fly.zip" -d {TRAIN_PATH}

Определим список классов и их число:

# Определение списка имен классов
CLASS_LIST = sorted(os.listdir(TRAIN_PATH))

# Определение количества классов
CLASS_COUNT = len(CLASS_LIST)

# Проверка результата
print(f'Количество классов: {CLASS_COUNT}, метки классов: {CLASS_LIST}')

# Отразим исходные изображения с курсовой камеры
# Создание заготовки для изображений всех классов
fig, axs = plt.subplots(1, CLASS_COUNT, figsize=(25, 5))

# Для всех номеров классов:
for i in range(CLASS_COUNT):
    # Формирование пути к папке содержимого класса
    pict_path = f'{TRAIN_PATH}{CLASS_LIST[i]}/'
    # Выбор случайного фото из i-го класса
    img_path = pict_path + random.choice(os.listdir(pict_path))
    # Отображение фотографии (подробнее будет объяснено далее)
    axs[i].set_title(CLASS_LIST[i])
    axs[i].imshow(Image.open(img_path))
    axs[i].axis('off')

# Отрисовка всего полотна
plt.show()

# Выделение области не содержащей данных о положении модели относительно горизонта

for dir_name in os.listdir(TRAIN_PATH):
  for file_name in os.listdir(TRAIN_PATH + dir_name):
    # Открытие и смена размера изображения
    img_path = TRAIN_PATH + dir_name + '/' + file_name
    img = Image.open(img_path).crop((IMG_LEFT, IMG_TOP, IMG_RIGHT, IMG_BOTTOM)).resize((IMG_HEIGHT, IMG_WIDTH))
    img.save(img_path)

///Исходная база ещё не разделена на выборки. Вполне естественно, что для тестовых данных аугментация не требуется - они не участвуют в обучении модели. По этой причине данные для теста необходимо сразу отделить от обучающих и проверочных. Таким образом, часть изображений просто выделяется для тестирования модели и размещается в отдельной папке. Для разделения данных требуется не только создать папку и указать путь к ней; также нужно определить количество изображений в каждом из трёх классов, выделить некоторую их долю (заданную в гиперпараметрах) в каждом классе, и уже потом переместить файлы по указанному пути в папку. Разумеется, правильнее решить эту задачу при помощи цикла:///

# Перенос файлов для теста в отдельное дерево папок, расчет размеров наборов данных

os.mkdir(TEST_PATH)                                        # Создание папки для тестовых данных
train_count = 0
test_count = 0

for class_name in CLASS_LIST:                              # Для всех классов по порядку номеров (их меток)
    class_path = f'{TRAIN_PATH}/{class_name}'              # Формирование полного пути к папке с изображениями класса
    test_path = f'{TEST_PATH}/{class_name}'                # Полный путь для тестовых данных класса
    class_files = os.listdir(class_path)                   # Получение списка имен файлов с изображениями текущего класса
    class_file_count = len(class_files)                    # Получение общего числа файлов класса
    os.mkdir(test_path)                                    # Создание подпапки класса для тестовых данных
    test_file_count = int(class_file_count * TEST_SPLIT)   # Определение числа тестовых файлов для класса
    test_files = class_files[-test_file_count:]            # Выделение файлов для теста от конца списка
    for f in test_files:                                   # Перемещение тестовых файлов в папку для теста
        os.rename(f'{class_path}/{f}', f'{test_path}/{f}')
    train_count += class_file_count                        # Увеличение общего счетчика файлов обучающего набора
    test_count += test_file_count                          # Увеличение общего счетчика файлов тестового набора

    print(f'Размер класса {class_name}: {class_file_count} машин, для теста выделено файлов: {test_file_count}')

print(f'Общий размер базы: {train_count}, выделено для обучения: {train_count - test_count}, для теста: {test_count}')

## Аугментация и формирование выборок

///Главная задача по увеличению базы изображений для обучения решается путем использования встроенного механизма Keras, он называется **ImageDataGenerator**. В параметрах генератора задаются все необходимые разрешения и пределы изменений исходного изображения - сдвиг, вращение, увеличение или отдаление, яркость и другие. Там же можно произвести нормализацию данных, что очень удобно (за это отвечает параметр **rescale**). Все параметры генератора называются соответсвующим их действию образом; удобства ради значения для них были заданы одноименно в верхнем регистре (ячейка гиперпараметров в начале ноутбука). Теперь остаётся только создать отдельный генератор для обучения, и отдельный для теста модели.///

### Генераторы изображений и выборок

# Генераторы изображений

# Изображения для обучающего набора нормализуются и аугментируются согласно заданным гиперпараметрам
# Далее набор будет разделен на обучающую и проверочную выборку в соотношении VAL_SPLIT
train_datagen = ImageDataGenerator(
                    rescale=1. / 255.,
                    rotation_range=ROTATION_RANGE,
                    width_shift_range=WIDTH_SHIFT_RANGE,
                    height_shift_range=HEIGHT_SHIFT_RANGE,
                    zoom_range=ZOOM_RANGE,
                    brightness_range=BRIGHTNESS_RANGE,
                    horizontal_flip=HORIZONTAL_FLIP,
                    validation_split=VAL_SPLIT
                )

# Изображения для тестового набора только нормализуются
test_datagen = ImageDataGenerator(
                   rescale=1. / 255.
                )

///Изображения для теста должны остаться в исходном виде. С ними будет проведена только нормализация.

Существует удобный метод генератора - `.flow_from_directory()`, который помогает извлечь из папок изображения для генерации, посчитать классы и автоматически вычислить метки классов для изображений. Было создано два отдельных генератора, поэтому тестовые изображения при прохождении через генератор не аугментируются, а только нормализуются. При этом обучающая и проверочная выборки будет иметь кол-во элементов в одном подаваемом объекте - BATCH_SIZE(24), а тестовая выборка включит в себя все тестовые изображения (batch_size равен test_count(341)), когда всего тестовых изображение тоже test_count((341))).///

# Обучающая выборка генерируется из папки обучающего набора
train_generator = train_datagen.flow_from_directory(
    # Путь к обучающим изображениям
    TRAIN_PATH,
    # Параметры требуемого размера изображения
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    # Размер батча
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    # Указание сгенерировать обучающую выборку
    subset='training'
)

# Проверочная выборка также генерируется из папки обучающего набора
validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    # Указание сгенерировать проверочную выборку
    subset='validation'
)

# Тестовая выборка генерируется из папки тестового набора
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=test_count,
    class_mode='categorical',
    shuffle=True,
)

# Проверка формы данных
print(f'Формы данных тренировочной выборки: {train_generator[0][0].shape}, {train_generator[0][1].shape}, батчей: {len(train_generator)}')
print(f'Формы данных   проверочной выборки: {validation_generator[0][0].shape}, {validation_generator[0][1].shape}, батчей: {len(validation_generator)}')
print(f'Формы данных      тестовой выборки: {test_generator[0][0].shape}, {test_generator[0][1].shape}, батчей: {len(test_generator)}')

print()

# Проверка назначения меток классов
print(f'Метки классов тренировочной выборки: {train_generator.class_indices}')
print(f'Метки классов   проверочной выборки: {validation_generator.class_indices}')
print(f'Метки классов      тестовой выборки: {test_generator.class_indices}')

### Проверка работы генераторов выборок

///Посмотрите, как работают генераторы выборок. Для начала позапускайте следующую ячейку несколько раз. Аугментация работает "на лету".

Индексы для примера выбраны такие: `[1][0][2]`, что означает:
 - **1**: номер батча в выборке. Сам батч - это кортеж (**x_train, y_train**) с примерами и метками классов;
 - **0**: первый элемент в кортеже, то есть сами изображения. Под индексом **1** будут метки классов в формате **one hot encoding** для изображений;
 - **2**: номер картинки в батче.///

# Проверка одного изображения из выборки
plt.imshow(train_generator[1][0][2])
plt.show()

#Сама же картинка представлена в виде трехмерного массива нормализованных пикселей (от **0** до **1**):

train_generator[1][0][2]

///Теперь создадим функцию для удобного просмотра сразу множества картинок из заданного батча. Для отрисовки нескольких изображений используем функцию `.subplots()` библиотеки **matplolib.pyplot**.///



# Функция рисования образцов изображений из заданной выборки

def show_batch(batch,                # батч с примерами
               img_range=range(20),  # диапазон номеров картинок
               figsize=(25, 8),      # размер полотна для рисования одной строки таблицы
               columns=5             # число колонок в таблице
               ):

    for i in img_range:
        ix = i % columns
        if ix == 0:
            fig, ax = plt.subplots(1, columns, figsize=figsize)
        class_label = np.argmax(batch[1][i])
        ax[ix].set_title(CLASS_LIST[class_label])
        ax[ix].imshow(batch[0][i])
        ax[ix].axis('off')
        plt.tight_layout()

    plt.show()

///Посмотрим на примеры картинок из генераторов. Можно запускать ячейки несколько раз - аугментация работает "на лету".///

## Создание и обучение модели нейронной сети

///После того, как вы убедились в правильной работе объекта **ImageDataGenerator** на примерах из конкретного батча, можно перейти к обучению модели. Для этого будет удобнее сразу создать необходимые функции, включающие компиляцию, обучение, построение графиков ошибки и точности, оценку предсказаний модели. Впоследствие функции можно использовать многократно, избегая повторов и внесения ошибок в код.
///

###Сервисные функции


# Функция компиляции и обучения модели нейронной сети
# По окончанию выводит графики обучения

def compile_train_model(model,                  # модель нейронной сети
                        train_data,             # обучающие данные
                        val_data,               # проверочные данные
                        optimizer=OPTIMIZER,    # оптимизатор
                        epochs=EPOCHS,          # количество эпох обучения
                        batch_size=BATCH_SIZE,  # размер батча
                        figsize=(20, 5)):       # размер полотна для графиков

    # Компиляция модели
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Вывод сводки
    model.summary()

    # Обучение модели с заданными параметрами
    history = model.fit(train_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=val_data)

    # Вывод графиков точности и ошибки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('График процесса обучения модели')
    ax1.plot(history.history['accuracy'],
               label='Доля верных ответов на обучающем наборе')
    ax1.plot(history.history['val_accuracy'],
               label='Доля верных ответов на проверочном наборе')
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('Эпоха обучения')
    ax1.set_ylabel('Доля верных ответов')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='Ошибка на обучающем наборе')
    ax2.plot(history.history['val_loss'],
               label='Ошибка на проверочном наборе')
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('Эпоха обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()
    plt.show()

# Функция вывода результатов оценки модели на заданных данных

def eval_model(model,
               x,                # данные для предсказания модели (вход)
               y_true,           # верные метки классов в формате OHE (выход)
               class_labels=[],  # список меток классов
               cm_round=3,       # число знаков после запятой для матрицы ошибок
               title='',         # название модели
               figsize=(10, 10)  # размер полотна для матрицы ошибок
               ):
    # Вычисление предсказания сети
    y_pred = model.predict(x)
    # Построение матрицы ошибок
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')
    # Округление значений матрицы ошибок
    cm = np.around(cm, cm_round)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Нейросеть {title}: матрица ошибок нормализованная', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    ax.images[-1].colorbar.remove()       # Стирание ненужной цветовой шкалы
    fig.autofmt_xdate(rotation=45)        # Наклон меток горизонтальной оси
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    plt.show()

    print('-'*100)
    print(f'Нейросеть: {title}')

    # Для каждого класса:
    for cls in range(len(class_labels)):
        # Определяется индекс класса с максимальным значением предсказания (уверенности)
        cls_pred = np.argmax(cm[cls])
        # Формируется сообщение о верности или неверности предсказания
        msg = 'ВЕРНО :-)' if cls_pred == cls else 'НЕВЕРНО :-('
        # Выводится текстовая информация о предсказанном классе и значении уверенности
        print('Класс: {:<20} {:3.0f}% сеть отнесла к классу {:<20} - {}'.format(class_labels[cls],
                                                                               100. * cm[cls, cls_pred],
                                                                               class_labels[cls_pred],
                                                                               msg))

    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    print('\nСредняя точность распознавания: {:3.0f}%'.format(100. * cm.diagonal().mean()))

# Совместная функция обучения и оценки модели нейронной сети

def compile_train_eval_model(model,                    # модель нейронной сети
                             train_data,               # обучающие данные
                             val_data,                 # проверочные данные
                             test_data,                # тестовые данные
                             class_labels=CLASS_LIST,  # список меток классов
                             title='',                 # название модели
                             optimizer=OPTIMIZER,      # оптимизатор
                             epochs=EPOCHS,            # количество эпох обучения
                             batch_size=BATCH_SIZE,    # размер батча
                             graph_size=(20, 5),       # размер полотна для графиков обучения
                             cm_size=(10, 10)          # размер полотна для матрицы ошибок
                             ):

    # Компиляция и обучение модели на заданных параметрах
    # В качестве проверочных используются валидационные данные
    compile_train_model(model,
                        train_data,
                        val_data,
                        optimizer=optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        figsize=graph_size)

    # Вывод результатов оценки работы модели на тестовых данных
    eval_model(model, test_data[0][0], test_data[0][1],
               class_labels=class_labels,
               title=title,
               figsize=cm_size)

###Архитектура модели нейронной сети


#Теперь создадим модель, обучим ее на генерируемых данных и оценим работу на тестовых:

# Создание модели последовательной архитектуры
model = Sequential()

# 1 сверточный блок
model.add(Conv2D(256, (4, 4), name='First_C', padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

# 2 сверточный блок
model.add(Conv2D(256, (4, 4), name='Second_C', padding='same', dilation_rate=(4, 4), activation='relu'))
model.add(Dropout(0.3, name='Second_D'))

# 3 сверточный блок
model.add(Conv2D(256, (4, 4), name='Third_C', padding='valid', dilation_rate=(4, 4), activation='relu'))
model.add(Dropout(0.3, name='Third_D'))

# 4 сверточный блок
model.add(Conv2D(256, (4, 4), name='Fourth_C', padding='valid', dilation_rate=(4, 4) , activation='relu'))
model.add(Dropout(0.3, name='Fourth_D'))

# Блок классификации
model.add(Flatten(name='Class_1'))
model.add(Dense(128, activation='relu', name='Class_2'))
model.add(Dense(256, activation='relu', name='Class_3'))
model.add(Dense(CLASS_COUNT, activation='softmax', name='Class_4'))

# Обучение модели и вывод оценки ее работы на тестовых данных


compile_train_eval_model(model,
                         train_generator,
                         validation_generator,
                         test_generator,
                         class_labels=CLASS_LIST,
                         title='Сверточный классификатор')

compile_train_eval_model(model,
                         train_generator,
                         validation_generator,
                         test_generator,
                         class_labels=CLASS_LIST,
                         title='Сверточный классификатор')

# Для отрисовки изображений
from tensorflow.keras.preprocessing import image

# Формирование пути к папке содержимого класса
img_path = f'{TEST_PATH}{CLASS_LIST[0]}/'
img_pred = img_path + random.choice(os.listdir(img_path))

import matplotlib.image as mpimg
plt.imshow(mpimg.imread(img_pred))

# Отрисовка всего полотна
plt.show()

xtest =[]

print(img_pred)

# Открытие картинки и изменение ее размера для соответсвия входу модели
img = Image.open(img_pred).resize((IMG_HEIGHT, IMG_WIDTH))
# Преобразование картинки в numpy-массив чисел с плавающей запятой и нормализация значений пикселей
image = np.array(img, dtype='float64') / 255.

# добавление оси для совпадения формы входа модели; получается батч из одного примера
image = np.expand_dims(image, axis=0)

# Распознавание изображения нейросетью
pred = model.predict(image)
print('Результат распознавания:')

print(pred)

##Подведем итоги

///Провели обучение модели на основе базового метода `.fit()`. Эта функция совместно с генераторами данных и выборок позволила применить различные виды аугментации для формирования каждый раз новых батчей с измененными изображениями. В итоге та же модель, которую вы обучали ранее без аугментации, может учиться глубже и лучше.

За весь период обучения вы подавали намного больше данных, аугментируя изображения для обучения на каждом шаге.

Удобство генератора в том, что создание аугментированных изображений на лету занимает память только на один батч. Если попробовать сразу вычислить всю большую аугментированную базу, памяти может просто не хватить.

На проверку модели на каждой эпохе вы подавали каждый раз обновленные версии проверочного набора. Как показывают графики, модель лучше с ними справлялась.

В финале оценили работу модели на тестовых данных, которые ей никогда не предъявлялись.///

