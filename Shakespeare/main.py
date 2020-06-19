from Server import Server
from Client import Client
from Model import Model
from preparation import Shakespeare
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
                    help="Set the learning rate",
                    type=float, default=0.0005)
parser.add_argument("-p", "--permutation",
                    help="Set the permutation index for aggregation",
                    type=int, default=0)
parser.add_argument("-g", "--gpu",
                    help="Set the GPU device",
                    type=int, default=None)
parser.add_argument("-c", "--char_dim",
                    help="Embedded characters' dimension",
                    type=int, default=8)

args = parser.parse_args()


if args.gpu is not None:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
else:
    tf.config.experimental.set_visible_devices(devices=[], device_type='GPU')
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

local_lr = args.lr

sentence_len = 80
char_dim = args.char_dim # embedding size of each character
train, test, char_indexes, indexes_char = Shakespeare(sentence_len)
n_chars = len(char_indexes.keys())

B = 500 #local batch size
E = 5 #epochs
C = 0.1 #fraction of clients - global batch size (=1 --> full batch gradient descent)
K = len(train) #n_clients

permutation = args.permutation


server = Server(Model(sentence_len, char_dim, n_chars, local_lr), B, E, C, K, local_lr, permutation)

clients = []
shared_clients_model = Model(sentence_len, char_dim, n_chars, local_lr)
print("Creating " + str(K) + " clients...")
for k in range(K):
    clients.append(Client(shared_clients_model, train=train[k], test=test[k]))


server.train(clients, rounds=100)
