# БЛОК НАСТРОЙКИ ГРАФИЧЕСКОГО ВЫВОДА
# Используется режим Agg, чтобы графики сохранялись в файлы,
# а не открывались отдельным окном.
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

# БЛОК ПОДКЛЮЧЕНИЯ БИБЛИОТЕК
# make_blobs используется для генерации тестового набора данных.
from sklearn.datasets import make_blobs

# KMeans используется для кластеризации методом K-средних.
from sklearn.cluster import KMeans

# KElbowVisualizer используется для поиска оптимального количества кластеров.
from yellowbrick.cluster import KElbowVisualizer

# Counter используется для подсчета количества объектов в каждом кластере.
from collections import Counter

# pandas используется для представления данных в табличном виде.
import pandas as pd

# matplotlib и seaborn используются для построения графиков.
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка шрифта, чтобы убрать повторяющиеся предупреждения findfont.
plt.rcParams['font.family'] = 'DejaVu Sans'


# БЛОК ГЕНЕРАЦИИ ДАННЫХ
# Формируется набор из 200 объектов с двумя признаками.
# centers=4 означает, что данные заранее распределены вокруг 4 центров.
dataset, classes = make_blobs(
    n_samples=200,
    n_features=2,
    centers=4,
    cluster_std=0.5,
    random_state=0
)

# БЛОК ПРЕОБРАЗОВАНИЯ ДАННЫХ
# Массив данных преобразуется в таблицу DataFrame
# с двумя столбцами: var1 и var2.
df = pd.DataFrame(dataset, columns=['var1', 'var2'])

print("Первые строки набора данных:")
print(df.head())


# БЛОК ПОИСКА ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ
# Создается модель KMeans без заранее заданного количества кластеров.
# Метод локтя проверяет значения k от 1 до 12.
model = KMeans(random_state=0, n_init=10)

visualizer = KElbowVisualizer(
    model,
    k=(1, 12)
)

# Выполняется обучение визуализатора на наборе данных.
visualizer.fit(df)

# График метода локтя сохраняется в файл elbow.png.
visualizer.show(outpath="elbow.png")

print("\nГрафик метода локтя сохранен в файл elbow.png")


# БЛОК КЛАСТЕРИЗАЦИИ МЕТОДОМ K-СРЕДНИХ
# По результатам метода локтя задается количество кластеров n_clusters=4.
# init='k-means++' используется для более качественного выбора начальных центров.
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    random_state=0,
    n_init=10
)

# Выполняется обучение модели KMeans.
kmeans.fit(df)


# БЛОК ВЫВОДА РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ
# labels_ содержит номер кластера для каждой точки данных.
print("\nМетки кластеров для каждой точки:")
print(kmeans.labels_)

# cluster_centers_ содержит координаты центроидов кластеров.
print("\nКоординаты центроидов:")
print(kmeans.cluster_centers_)

# inertia_ содержит внутрикластерную сумму квадратов.
print("\nВнутрикластерная сумма квадратов:")
print(kmeans.inertia_)

# n_iter_ показывает количество итераций,
# которое потребовалось алгоритму для сходимости.
print("\nКоличество итераций:")
print(kmeans.n_iter_)

# Counter считает количество объектов в каждом кластере.
print("\nРазмер каждого кластера:")
print(Counter(kmeans.labels_))


# БЛОК ВИЗУАЛИЗАЦИИ РЕЗУЛЬТАТОВ
# Создается точечная диаграмма распределения объектов по кластерам.
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x='var1',
    y='var2',
    hue=kmeans.labels_
)

plt.title('Кластеризация методом KMeans')
plt.xlabel('var1')
plt.ylabel('var2')
plt.legend(title='Кластер')

# Диаграмма сохраняется в файл scatter.png.
plt.savefig("scatter.png")

print("\nТочечная диаграмма сохранена в файл scatter.png")