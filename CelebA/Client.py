import numpy as np
from copy import deepcopy
import pandas as pd

class Client:
    def __init__(self, model, train, test):
        self.model = model
        self.X = np.array(train[0])
        self.y = np.array(train[2])

        self.X_test = np.array(test[0])
        self.y_test = np.array(test[2])

        self.imaage_sharpness = np.where(np.array(train[1]) == 0)[0].size/self.X.shape[0]  # rate of not blurry (so sharp) images
        self.datasetSize = self.X.shape[0]

        a = np.where(self.y == 0)[0].size
        b = np.where(self.y == 1)[0].size
        self.classes_balancing = min(a,b)/max(a,b)

        self.testSize = self.X_test.shape[0]

        # self.weights_list = None

    def update(self, B, E, W):
        self.model.set_weights(W)
        if B is None:
            B = self.datasetSize
        self.model.fit(x=self.X, y=self.y, verbose=0, batch_size=B, epochs=E)

        # self.weights_list = []
        # for layer in range(len(self.model.layers)):
        #     try:
        #         self.weights_list.append((deepcopy(self.model.layers[layer].get_weights()[0]),
        #                                   deepcopy(self.model.layers[layer].get_weights()[1])))
        #     except:
        #         pass


    def evaluate(self, W):
        self.model.set_weights(W)
        _, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return accuracy

    def gen_stats(self, path, client_index):
        y_pred = self.model.predict(self.X_test)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        df = pd.DataFrame(data={'y_pred': y_pred.flatten(), 'y_true': self.y_test})
        path += "client_" + str(client_index+1) + ".csv"
        df.to_csv(path, index=False)


