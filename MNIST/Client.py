import numpy as np
from copy import deepcopy
import pandas as pd

class Client:
    def __init__(self, model, X, y, X_test, y_test):
        self.model = model
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.datasetSize = X.shape[0]
        self.classes = np.unique(y, axis=0).size

        self.testSize = X_test.shape[0]

    def update(self, B, E, W):
        self.model.set_weights(deepcopy(W))
        if B is None:
            B = self.datasetSize
        self.model.fit(x=self.X, y=self.y, verbose=0, batch_size=B, epochs=E)

    def evaluate(self, W):
        self.model.set_weights(deepcopy(W))
        _, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return accuracy

    def gen_stats(self, path, client_index):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(self.y_test, axis=1)
        df = pd.DataFrame(data={'y_pred': y_pred, 'y_true': y_test})
        path += "client_" + str(client_index+1) + ".csv"
        df.to_csv(path, index=False)


