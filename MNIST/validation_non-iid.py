from Server import Server
from Client import Client
from Model import Model
from preparation import MNIST
import tensorflow as tf
import sys
import os
import pandas as pd
from datetime import datetime

n_gpu = int(sys.argv[1])
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[n_gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[n_gpu], True)

B = None #local batch size
E = 5 #epochs
C = 0.1 #fraction of clients - global batch size (=1 --> full batch gradient descent)
K = 100 #n_clients
iid = False

train, validation = MNIST(iid=iid, validation = True)

lr_list = [0.5, 0.215, 0.1]
# lr_list = [0.05, 0.0215, 0.01]
target_acc = 0.95
target_perc = 50
max_round = 100

validation_dict = {}
path = "experiments/validation/" + str(datetime.now().strftime("%H%M%S_%d%b%Y")) + "/"
if not os.path.exists(path):
    os.makedirs(path)

for local_lr in lr_list:
    print("Learning rate = " + str(local_lr))

    server = Server(Model(local_lr), B, E, C, K, iid, local_lr)

    clients = [Client(Model(local_lr), X=train[k][0], y=train[k][1], X_test=validation[k][0], y_test=validation[k][1]) for k in range(K)]

    validation_dict[local_lr] = server.validate(clients, target_acc, target_perc, max_round)

    del server
    del clients

df = pd.DataFrame(data={'learning_rate': list(validation_dict.keys()), 'rounds': list(validation_dict.values())})
df.to_csv(path + "results.csv", index=False)

with open(path + 'params.txt', 'w+') as file:
    file.write("iid: " + str(iid) + "\r\n")
    file.write("target_acc: " + str(target_acc) + "\r\n")
    file.write("target_perc: " + str(target_perc) + "\r\n")
    file.write("max round: " + str(max_round))