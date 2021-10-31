import src.data.load_and_prepare as dm
import src.models.create_train_predict as mm

# Обработка сырых тренировочных и тестовых данных
train = dm.prepare_train_data()
test = dm.prepare_test_data()

mm.create_rf_model()
mm.make_prediction(test)

# Оставшийся pipeline IN PROGRESS
# Смотри notebooks/Titanic.ipynb
