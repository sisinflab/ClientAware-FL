import numpy as np
from copy import deepcopy
import pandas as pd

class Client:
    def __init__(self, model, train, test):
        self.model = model
        self.X = np.array(train[0])
        self.y = np.array(train[1])

        self.X_test = np.array(test[0])
        self.y_test = np.array(test[1])

        self.datasetSize = self.X.shape[0]
        self.testSize = self.X_test.shape[0]

        # X_chars = set()
        # for sentence in self.X:
        #     for char in sentence:
        #         X_chars = X_chars.union(set(np.where(char == True)[0]))
        # self.X_train_diversity = len(X_chars)/self.X.shape[2]
        #
        # y_chars = set()
        # for char in self.y:
        #     y_chars = y_chars.union(set(np.where(char == True)[0]))
        # self.y_train_diversity = len(y_chars)/self.y.shape[1]


    def update(self, B, E, W):
        self.model.set_weights(W)
        if B is None:
            B = self.datasetSize
        self.model.fit(x=self.X, y=self.y, verbose=0, batch_size=B, epochs=E)

    def evaluate(self, W):
        self.model.set_weights(W)
        _, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return accuracy

    def gen_stats(self, path, client_index):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(self.y_test, axis=1)
        df = pd.DataFrame(data={'y_pred': y_pred, 'y_true': y_test})
        path += "client_" + str(client_index+1) + ".csv"
        df.to_csv(path, index=False)


