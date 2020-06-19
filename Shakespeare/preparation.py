import json
import numpy as np
import pickle
import os

def Shakespeare(sentence_len):
    path = "train_08_test_02_senlen_" + str(sentence_len)
    if os.path.exists(path):
        print("Finded a previous dataset version. Loading...")
        with open(path, 'rb') as file:
            return pickle.load(file)


    source_path = "../../leaf/data/shakespeare/data/"
    # source_path = ""
    with open(source_path + 'train/all_data_niid_0_keep_0_train_9.json', 'r') as f:
        file_train = json.load(f)

    with open(source_path + 'test/all_data_niid_0_keep_0_test_9.json', 'r') as f:
        file_test = json.load(f)

    train = []
    test = []

    chars = set()
    for user in file_train['users']:

        for sentence in file_train['user_data'][user]['x']: # list of sentences
            chars = chars.union(set(sentence.lower()))

        for sentence in file_train['user_data'][user]['y']:
            chars = chars.union(set(sentence.lower()))

        for sentence in file_test['user_data'][user]['x']:
            chars = chars.union(set(sentence.lower()))

        for sentence in file_test['user_data'][user]['y']:
            chars = chars.union(set(sentence.lower()))

    chars = sorted(list(chars))
    chars_enc = np.identity(len(chars), dtype=int)

    chars_one_hot = dict((chars[i],chars_enc[i,:]) for i in range(len(chars)))

    char_indexes = dict((c, i) for i, c in enumerate(chars))
    indexes_char = dict((i, c) for i, c in enumerate(chars))

    for user in file_train['users']:
        # if len(file_train['user_data'][user]['x']) > 20000:
        #     continue

        print("Generating dataset for client " + str(len(train) + 1))

        i_sentences_train = [sentence.lower() for sentence in file_train['user_data'][user]['x']] # list of sentences in lowercase
        i_chars_train = [char.lower() for char in file_train['user_data'][user]['y']]  # list of chars in lowercase
        i_sentences_test = [sentence.lower() for sentence in file_test['user_data'][user]['x']]
        i_chars_test = [char.lower() for char in file_test['user_data'][user]['y']]

        i_x_train = np.zeros((len(i_sentences_train), sentence_len), dtype=int)
        i_y_train = []
        i_x_test = np.zeros((len(i_sentences_test), sentence_len), dtype=int)
        i_y_test = []

        for i, sentence in enumerate(i_sentences_train):
            for t, char in enumerate(sentence):
                i_x_train[i , t] = char_indexes[char]
            i_y_train.append(chars_one_hot[i_chars_train[i]])

        for i, sentence in enumerate(i_sentences_test):
            for t, char in enumerate(sentence):
                i_x_test[i, t] = char_indexes[char]
            i_y_test.append(chars_one_hot[i_chars_test[i]])

        #normalization
        i_x_train = i_x_train / float(len(chars))
        i_x_test = i_x_test / float(len(chars))


        train.append((i_x_train, np.array(i_y_train)))
        test.append((i_x_test, np.array(i_y_test)))

        # if len(train) == 10:
        #     break


    print("Saving training and test set...")
    with open(path,"wb") as file:
        pickle.dump((train, test, char_indexes, indexes_char), file)
    return train, test, char_indexes, indexes_char




# def Shakespeare(sentence_len, char_dim):
#     path = "train_08_test_02_senlen_" + str(sentence_len) + "chardim_" + str(char_dim)
#     if os.path.exists(path):
#         print("Finded a previous dataset version. Loading...")
#         with open(path, 'rb') as file:
#             return pickle.load(file)
#
#
#     source_path = "../../leaf/data/shakespeare/data/"
#     # source_path = ""
#     with open(source_path + 'train/all_data_niid_0_keep_0_train_9.json', 'r') as f:
#         file_train = json.load(f)
#
#     with open(source_path + 'test/all_data_niid_0_keep_0_test_9.json', 'r') as f:
#         file_test = json.load(f)
#
#     train = []
#     test = []
#
#     chars = set()
#     for user in file_train['users']:
#
#         for sentence in file_train['user_data'][user]['x']: # list of sentences
#             chars = chars.union(set(sentence.lower()))
#
#         for sentence in file_train['user_data'][user]['y']:
#             chars = chars.union(set(sentence.lower()))
#
#         for sentence in file_test['user_data'][user]['x']:
#             chars = chars.union(set(sentence.lower()))
#
#         for sentence in file_test['user_data'][user]['y']:
#             chars = chars.union(set(sentence.lower()))
#
#     chars = sorted(list(chars))
#     chars_enc = np.identity(len(chars))
#     emb_chars = np.array(PCA(char_dim).fit_transform(chars_enc), dtype='float32')
#     chars_map = dict((chars[i], emb_chars[i,:]) for i in range(len(chars)))
#
#     char_indexes = dict((c, i) for i, c in enumerate(chars))
#     indexes_char = dict((i, c) for i, c in enumerate(chars))
#
#     for user in file_train['users']:
#         print("Generating dataset for client " + str(len(train) + 1))
#
#         i_sentences_train = [sentence.lower() for sentence in file_train['user_data'][user]['x']] # list of sentences in lowercase
#         i_chars_train = [char.lower() for char in file_train['user_data'][user]['y']]  # list of chars in lowercase
#         i_sentences_test = [sentence.lower() for sentence in file_test['user_data'][user]['x']]
#         i_chars_test = [char.lower() for char in file_test['user_data'][user]['y']]
#
#         # one hot encoding: x is a 3D matrix with dim: n_senteces x sentence_len x char_dim
#         #                   y is a 2D matrix with dim: n_y_chars x char_dim
#         i_x_train = np.zeros((len(i_sentences_train), sentence_len, char_dim), dtype='float32')
#         i_y_train = np.zeros((len(i_chars_train), len(chars)), dtype=int)
#         i_x_test = np.zeros((len(i_sentences_test), sentence_len, char_dim), dtype='float32')
#         i_y_test = np.zeros((len(i_chars_test), len(chars)), dtype=int)
#
#
#         for i, sentence in enumerate(i_sentences_train):
#             for t, char in enumerate(sentence):
#                 i_x_train[i, t, :] = chars_map[char]
#             i_y_train[i, char_indexes[i_chars_train[i]]] = 1
#
#         for i, sentence in enumerate(i_sentences_test):
#             for t, char in enumerate(sentence):
#                 i_x_test[i, t, :] = chars_map[char]
#             i_y_test[i, char_indexes[i_chars_test[i]]] = 1
#
#         train.append((i_x_train, i_y_train))
#         test.append((i_x_test, i_y_test))
#
#         # if len(train) == 10:
#         #     break
#
#     print("Saving training and test set...")
#     with open(path,"wb") as file:
#         pickle.dump((train, test, char_indexes, indexes_char), file)
#     return train, test, char_indexes, indexes_char
