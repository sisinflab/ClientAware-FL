import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

def MNIST(iid, validation = False):
    n_clients = 100
    num_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if validation:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    test_disp = int(X_test.shape[0]/n_clients)

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    train_list = []
    test_list = []

    if iid:
        train_disp = int(X_train.shape[0]/n_clients)
        train = list(zip(X_train, y_train))
        np.random.shuffle(train)
        X_train, y_train = zip(*train)

        for i in range(n_clients):
            i_X_train = X_train[i * train_disp:(i + 1) * train_disp]
            i_y_train = y_train[i * train_disp:(i + 1) * train_disp]

            i_X_test = X_test[i * test_disp:(i + 1) * test_disp]
            i_y_test = y_test[i * test_disp:(i + 1) * test_disp]

            train_list.append((np.array(i_X_train), to_categorical(i_y_train, num_classes)))
            test_list.append((np.array(i_X_test), to_categorical(i_y_test, num_classes)))


    else: #non-iid
        idx = np.argsort(y_train)
        train = list(zip(X_train[idx], y_train[idx]))

        shard_size = int((X_train.shape[0]/n_clients)/2)
        n_shards = 200
        shards = [train[i * shard_size:(i + 1) * shard_size] for i in range(n_shards)]
        np.random.shuffle(shards)

        for i in range(n_clients):
            i_client_shards = shards.pop() + shards.pop()
            i_X_train, i_y_train = zip(*i_client_shards)

            i_X_test = X_test[i * test_disp:(i + 1) * test_disp]
            i_y_test = y_test[i * test_disp:(i + 1) * test_disp]

            train_list.append((np.array(i_X_train), to_categorical(i_y_train, num_classes)))
            test_list.append((np.array(i_X_test), to_categorical(i_y_test, num_classes)))

    return train_list, test_list