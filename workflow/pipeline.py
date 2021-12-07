import os, sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.dirname(cur_dir)
sys.path.append(par_dir)

import src.data.load_and_prepare as dm
import src.models.create_train_predict as mm
import click


# Обработка сырых тренировочных и тестовых данных
def prepare_data():
    train = dm.prepare_train_data()
    test = dm.prepare_test_data()
    return train, test


# Создание модели
def create_model():
    mm.create_rf_model()


# Создание предсказания
def full_pipeline():
    train, test = prepare_data()
    create_model()
    mm.make_prediction(test)


@click.command()
@click.option("--type", "-t",
              help=f"Runs pipeline in one of possible ways:\nprepare_data, create_model, full")
def main(type):
    """A small script for Titanic project"""
    if type == "prepare_data":
        prepare_data()
    elif type == "create_model":
        create_model()
    elif type == "full":
        full_pipeline()
    elif type is None:
        print("Hi, glad you here!\nIf you want this script to do something, please, use --type option")
    else:
        print(f"Hey, '{type}' is not possible option.\nUse --help to get more information")


if __name__ == "__main__":
    main()
