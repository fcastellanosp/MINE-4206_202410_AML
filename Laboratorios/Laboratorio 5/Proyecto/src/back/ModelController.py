import Definitions
import os.path as osp
import pandas as pd

from keras.models import load_model
from src.model.TextPreprocessing import TextPreprocessing


class ModelController:

    def __init__(self):
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models", "text_classifier.h5")
        self.model = load_model(self.model_path)
        # El número de componentes con el que fue entrenado el modelo
        self.n_components = 50
        self.t_processing = TextPreprocessing()

    def predict(self, text):
        print("predict ->")
        # Debemos preparar la información de la misma forma como preparamos el modelo
        x = self.t_processing.transform([text], self.n_components)
        y_pred_prob = self.model.predict(x, verbose=0)
        df = pd.DataFrame(self.get_categories(), columns=["Category"])
        df['Probability'] = y_pred_prob.reshape(-1, 1)

        return df

    def get_categories(self):
        return ["Entertainment", "Business", "Technology", "Sports", "Education"]
