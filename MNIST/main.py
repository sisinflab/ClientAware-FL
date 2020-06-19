from Server import Server
from Client import Client
from Model import Model
from preparation import MNIST
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--iid",
                    help="Set if you want IID distribution",
                    action='store_true')
parser.add_argument("--lr",
                    help="Set the learning rate",
                    type=float, default=None)
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

B = None #local batch size
E = 5 #epochs
C = 0.1 #fraction of clients - global batch size (=1 --> full batch gradient descent)
K = 100 #n_clients

iid = args.iid

if args.lr is None:
    if iid:
        local_lr = 0.215
    else:
        local_lr = 0.1
else:
    local_lr = args.lr

train, test = MNIST(iid=iid)
server = Server(Model(local_lr), B, E, C, K, iid, local_lr, args.permutation)

clients = [Client(Model(local_lr), X=train[k][0], y=train[k][1], X_test=test[k][0], y_test=test[k][1]) for k in range(K)]

server.train(clients, rounds=1000)

