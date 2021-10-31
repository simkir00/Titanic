import os
import pandas as pd

from definition import ROOT_DIR

RAW_TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "data/raw/train.csv")
RAW_TEST_DATA_PATH = os.path.join(ROOT_DIR, "data/raw/test.csv")

NEW_TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "data/processed/train.csv")
NEW_TEST_DATA_PATH = os.path.join(ROOT_DIR, "data/processed/test.csv")


def load_data(path):
    data = pd.read_csv(path, index_col=0)
    return data


def prepare_train_data():
    # Загрузка данных
    train_data = load_data(RAW_TRAIN_DATA_PATH)

    # Заполнение пропущенных значений места посадки
    train_data = train_data.fillna({"Embarked": "S"})

    # Заполнение пропущенных значений возраста
    train_data['Age'] = train_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

    # Baseline "фильтрация" фичей
    train_data = train_data.drop(["Name", "Cabin", "Ticket", "SibSp", "Parch"], axis=1)

    # One hot encoding преобразование категориальных фичей
    train_data = pd.get_dummies(train_data, columns=["Sex", "Pclass", "Embarked"])

    # Сохранение обработанного датасета
    train_data.to_csv(NEW_TRAIN_DATA_PATH)

    print("Train data were preprocessed!")
    return train_data


def prepare_test_data():
    # Загрузка данных
    test_data = load_data(RAW_TEST_DATA_PATH)

    # Заполнение пропущенных значений места посадки
    test_data = test_data.fillna({"Embarked": test_data.Embarked.mode()[0]})

    # Заполнение пропущенных значений возраста
    test_data['Age'] = test_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

    # Заполнение пропущенных значений цены билета
    med_fare = test_data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    test_data['Fare'] = test_data['Fare'].fillna(med_fare)

    # Baseline "фильтрация" фичей
    test_data = test_data.drop(["Name", "Cabin", "Ticket", "SibSp", "Parch"], axis=1)

    # One hot encoding преобразование категориальных фичей
    test_data = pd.get_dummies(test_data, columns=["Sex", "Pclass", "Embarked"])

    # Сохранение обработанного датасета
    test_data.to_csv(NEW_TEST_DATA_PATH)

    print("Test data were preprocessed!")
    return test_data
