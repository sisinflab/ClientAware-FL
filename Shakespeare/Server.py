from random import shuffle
import numpy as np
from datetime import datetime
import os
import pandas as pd
from csv import writer

class Server:
    def __init__(self, model, B, E, C, K, local_lr, permutation):
        self.model = model
        self.E = E
        self.B = B
        self.C = C
        self.K = K

        self.curr_time = None
        self.local_lr = local_lr
        self.permutation = permutation

    def validate(self, clients, target_acc, target_perc, max_round):

        curr_round = 0
        while True:
            curr_round += 1
            print("ROUND " + str(curr_round))
            self.__round_train(clients)

            accuracy, perc = self.evaluate(clients, target_acc)
            print("The " + str(perc) + "% of clients gained the target accuracy = " + str(target_acc))
            print("Aggregate accuracy gained: " + str(accuracy))
            if perc >= target_perc:
                print("Target percentage gained")
                break
            if curr_round >= max_round:
                print("STOP: Max round reached")
                curr_round = None
                break

        return curr_round

    def train(self, clients, rounds, target_acc = 0.5):
        self.curr_time = str(datetime.now().strftime("%H%M%S_%d%b%Y"))

        for i in range(rounds):
            print("ROUND " + str(i + 1))
            self.__round_train(clients)

            accuracy, perc = self.evaluate(clients, target_acc)
            print("The " + str(perc) + "% of clients gained the target accuracy = " + str(target_acc))
            print("Aggregate accuracy gained: " + str(accuracy))
            # if (i - 4) % 5 == 0:  # save stats each 5 rounds
            self.gen_stats(clients, i + 1, target_acc, perc, accuracy)

    def support(self, client):
        client.update(self.B, self.E, self.model.get_weights())


    def __round_train(self, clients):
        m = max(round(self.C * self.K), 1)
        shuffle(clients)

        # for k in range(m):  # consider a random subset S of dimension m
        #     clients[k].update(self.B, self.E, self.model.get_weights())  # clientUpdate

        totSize = 0
        # totX_div = 0
        # toty_div = 0
        for k in range(m):
            totSize += clients[k].datasetSize
            # totX_div += clients[k].X_train_diversity
            # toty_div += clients[k].y_train_diversity

        c1 = []
        c4 = []
        c5 = []
        for k in range(m):
            c1.append(clients[k].datasetSize / totSize)
            # c4.append(clients[k].X_train_diversity / totX_div)
            # c5.append(clients[k].y_train_diversity / toty_div)

        perm = [[c1], [c4], [c5], [c1, c4, c5], [c1, c5, c4], [c4, c1, c5], [c5, c1, c4], [c4, c5, c1], [c5, c4, c1]]
        p = self.__compute_p(perm[self.permutation])

        # p = self.__compute_p([c1]) # only dataset size is considered (DS)
        # p = self.__compute_p([c4]) # only images sharpness is considered (IS)
        # p = self.__compute_p([c5]) # only class balancing is considered (CB)
        # p = self.__compute_p([c1, c4, c5]) # DS, IS, CB
        # p = self.__compute_p([c1, c5, c4]) # DS, CB, IS
        # p = self.__compute_p([c4, c1, c5]) # IS, DS, CB
        # p = self.__compute_p([c5, c1, c4]) # CB, DS, IS
        # p = self.__compute_p([c4, c5, c1]) # IS, CB, DS
        # p = self.__compute_p([c5, c4, c1]) # CB, IS, DS

        w = [0 for _ in range(len(self.model.layers))]
        b = [0 for _ in range(len(self.model.layers))]
        z = [0 for _ in range(len(self.model.layers))]

        for k in range(m):
            # update server weights
            clients[k].update(self.B, self.E, self.model.get_weights())  # clientUpdate
            for layer in range(len(self.model.layers)):
                try:
                    w[layer] += p[k] * clients[k].model.layers[layer].get_weights()[0]
                except:
                    pass
                try:
                    b[layer] += p[k] * clients[k].model.layers[layer].get_weights()[1]
                except:
                    pass
                try:
                    z[layer] += p[k] * clients[k].model.layers[layer].get_weights()[2]
                except:
                    pass

        for layer in range(len(self.model.layers)):
            try:
                self.model.layers[layer].set_weights([w[layer], b[layer], z[layer]])
            except:
                try:
                    self.model.layers[layer].set_weights([w[layer], b[layer]])
                except:
                    try:
                        self.model.layers[layer].set_weights([w[layer]])
                    except:
                        pass




            # if w[layer] != 0 and b[layer] != 0 and z[layer] != 0:
            #     self.model.layers[layer].set_weights([w[layer], b[layer], z[layer]])
            # elif w[layer] != 0 and b[layer] != 0 and z[layer] == 0:
            #     self.model.layers[layer].set_weights([w[layer], b[layer]])
            # elif w[layer] != 0 and b[layer] == 0 and z[layer] == 0:
            #     self.model.layers[layer].set_weights(w[layer])
            # try:
            #     self.model.layers[layer].set_weights([w[layer], b[layer], z[layer]])
            # except:
            #     pass


    def __compute_p(self, c_list):
        c_list = np.array(c_list).T
        s = []
        m = max(round(self.C * self.K), 1)

        for k in range(m):
            s.append(np.sum(np.cumprod(c_list[k])))

        Z = sum(s)
        p = []
        for k in range(m):
            p.append(s[k] / Z)

        return p

    def evaluate(self, clients, target_acc):
        accuracy = 0
        tot_testSize = 0
        n_sat_clients = 0

        for client in clients:
            tot_testSize += client.testSize
            client_acc = client.evaluate(self.model.get_weights())
            accuracy += client.testSize * client_acc
            if client_acc >= target_acc:
                n_sat_clients += 1

        accuracy = accuracy/tot_testSize
        perc = (n_sat_clients*100)/self.K

        return accuracy, perc



    def gen_stats(self, clients, n_round, target_acc, perc, accuracy):

        path = "experiments/" + self.curr_time + "/" + "round_" + str(n_round) + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        if n_round == 1:
            with open("experiments/" + self.curr_time + "/" + 'params.txt', 'w+') as file:
                file.write("Local learning rate: " + str(self.local_lr) + "\r\n")
                file.write("E (local epochs): " + str(self.E) + "\r\n")
                file.write("B (local batch size): " + str(self.B) + "\r\n")
                file.write("C (fraction of clients): " + str(self.C) + "\r\n")
                file.write("K (number of clients): " + str(self.K) + "\r\n")
                file.write("Target accuracy: " + str(target_acc) + "\r\n")
                file.write("Permutation: " + str(self.permutation))

            df = pd.DataFrame(data={'n_round': [n_round], '%_clients': [perc], 'aggr_accuracy': [accuracy]})
            df.to_csv("experiments/" + self.curr_time + "/" + "accuracy.csv", index=False)
        else:
            with open("experiments/" + self.curr_time + "/" + "accuracy.csv", 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow([n_round, perc, accuracy])

        for k in range(self.K):
            clients[k].gen_stats(path, k)

    # def __round_train(self, clients):
    #     m = max(round(self.C * self.K), 1)
    #     shuffle(clients)
    #
    #     for k in range(m):  # consider a random subset S of dimension m
    #         clients[k].update(self.B, self.E, self.model.get_weights())  # clientUpdate
    #
    #     totSize = 0
    #     for k in range(m):
    #         totSize += clients[k].datasetSize
    #
    #     # update server weights
    #     for layer in range(len(self.model.layers)):
    #         try:
    #             w = 0
    #             b = 0
    #             for k in range(m):
    #                 w += (clients[k].datasetSize / totSize) * clients[k].model.layers[layer].get_weights()[0]
    #                 b += (clients[k].datasetSize / totSize) * clients[k].model.layers[layer].get_weights()[1]
    #             self.model.layers[layer].set_weights([w, b])
    #         except:
    #             pass


