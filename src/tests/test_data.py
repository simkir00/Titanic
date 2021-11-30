import os
import sys
import pytest

# Вспомогательная часть для корректоного импортирования модулей
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import data.load_and_prepare as dm


# Тестирование с генерацией большого количества тестов
# Построчная проверка обработанных данных на отсутствие пустых значений
n_train = len(dm.load_data(dm.RAW_TRAIN_DATA_PATH))
n_test = len(dm.load_data(dm.RAW_TEST_DATA_PATH))

part_1 = [(dm.prepare_train_data, i) for i in range(n_train)]
part_2 = [(dm.prepare_test_data, j) for j in range(n_test)]
params = part_1 + part_2


@pytest.mark.parametrize("loader, row", params)
def test_data_loaders(loader, row):
    # print(row)
    res = loader().iloc[row]
    nul_values = res.isnull().sum().sum()
    # print(nul_values)

    assert nul_values == 0
