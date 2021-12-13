import os
import pandas as pd
import src.data.load_and_prepare as dm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import pickle

cur_dir = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(cur_dir))

PREPROCESSED_TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "data/processed/train.csv")
PREPROCESSED_TEST_DATA_PATH = os.path.join(ROOT_DIR, "data/processed/test.csv")

MODEL_PATH = os.path.join(ROOT_DIR, "models/rf_model.pkl")

RESULT_PATH = os.path.join(ROOT_DIR, "data/results/prediction.csv")


def create_rf_model():
    train_data = dm.load_data(PREPROCESSED_TRAIN_DATA_PATH)
    X = train_data.drop(["Survived"], axis=1)
    y = train_data.Survived

    max_depth_values = range(1, 20)
    n_estimators_values = range(100, 500, 50)

    # Создадим классификатор на основе случайного леса и подберём гиперпараметры
    # Предположили дефолтные значения как лучшие
    best_max_depth = None
    best_n_estimators = 100

    best_score = 0

    for max_depth in max_depth_values:
        for n_estimators in n_estimators_values:
            # print(f"AHA: {max_depth} and {n_estimators}")

            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            mean_cross_val_score = cross_val_score(clf, X, y, cv=5).mean()

            if mean_cross_val_score > best_score:
                best_score = mean_cross_val_score
                # best_model = clf
                best_max_depth = max_depth
                best_n_estimators = n_estimators

    best_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth)
    pickle.dump(best_model, open(MODEL_PATH, 'wb'))

    print(f"RandomForest classifier created!")
    return best_model


def train_model():
    train_data = dm.load_data(PREPROCESSED_TRAIN_DATA_PATH)
    X = train_data.drop(["Survived"], axis=1)
    y = train_data.Survived

    clf = pickle.load(open(MODEL_PATH, 'rb'))
    clf.fit(X, y)

    print("RandomForest classifier trained!")

    return clf


def make_prediction(test_data):
    clf = train_model()
    predictions = clf.predict(test_data)
    predictions_output = pd.DataFrame({"PassengerId": test_data.index, "Survived": predictions})

    predictions_output.to_csv(RESULT_PATH, index=False)
    print("Done! We have predictions for test data!\nLook for them in \"data/results/\"")
