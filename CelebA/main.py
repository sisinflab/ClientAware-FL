from Server import Server
from Client import Client
from Model import Model
from preparation import CelebA
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
                    help="Set the learning rate",
                    type=float)
parser.add_argument("-p", "--permutation",
                    help="Set the permutation index for aggregation",
                    type=int)
parser.add_argument("-g", "--gpu",
                    help="Set the GPU device",
                    type=int, default=None)

args = parser.parse_args()


if args.gpu is not None:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

local_lr = args.lr
image_size = 64

train, test = CelebA(image_size)

B = None #local batch size
E = 5 #epochs
C = 0.1 #fraction of clients - global batch size (=1 --> full batch gradient descent)
K = len(train) #n_clients


server = Server(Model(local_lr, image_size), B, E, C, K, local_lr, args.permutation)

clients = []
shared_clients_model = Model(local_lr, image_size)
print("Creating " + str(K) + " clients...")
for k in range(K):
    clients.append(Client(shared_clients_model, train=train[k], test=test[k]))


server.train(clients, rounds=100)

