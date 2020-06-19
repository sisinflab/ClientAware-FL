from random import shuffle
import numpy as np
from datetime import datetime
import os
import pandas as pd
from csv import writer

class Server:
    def __init__(self, model, B, E, C, K, iid, local_lr, permutation):
        self.model = model
        self.E = E
        self.B = B
        self.C = C
        self.K = K

        self.curr_time = None
        self.iid = iid
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

    def train(self, clients, rounds, target_acc = 0.95):
        self.curr_time = str(datetime.now().strftime("%H%M%S_%d%b%Y"))

        for i in range(rounds):
            print("ROUND " + str(i + 1))
            self.__round_train(clients)

            accuracy, perc = self.evaluate(clients, target_acc)
            print("The " + str(perc) + "% of clients gained the target accuracy = " + str(target_acc))
            print("Aggregate accuracy gained: " + str(accuracy))
            #if (i - 4) % 5 == 0:  # save stats each 5 rounds
            self.gen_stats(clients, i + 1, target_acc, perc, accuracy)

    def __round_train(self, clients):
        m = max(round(self.C * self.K), 1)
        shuffle(clients)

        for k in range(m):  # consider a random subset S of dimension m
            clients[k].update(self.B, self.E, self.model.get_weights())  # clientUpdate

        phi = []
        totDelta = 0
        totSize = 0
        for k in range(m):
            norms = []
            totSize += clients[k].datasetSize
            totDelta += clients[k].classes
            for layer in range(len(self.model.layers)):
                try:  # used to skip layers without weights (pooling and flatten layers)
                    dif = self.model.layers[layer].get_weights()[0] - clients[k].model.layers[layer].get_weights()[0]
                    norms.append(np.linalg.norm(dif))
                except:
                    pass
                # b.append(self.model.layers[layer].get_weights()[1] - clients[k].model.layers[layer].get_weights()[1])
            phi.append(1 / np.sqrt(np.mean(norms) + 1))

        c1 = []
        c2 = []
        c3 = []
        phiTot = sum(phi)
        for k in range(m):
            c1.append(clients[k].datasetSize / totSize)
            c2.append(clients[k].classes / totDelta)
            c3.append(phi[k] / phiTot)
            
        perm = [[c1], [c2], [c3], [c1, c2, c3], [c1, c3, c2], [c2, c1, c3], [c3, c1, c2], [c2, c3, c1], [c3, c2, c1]]
        p = self.__compute_p(perm[self.permutation])
        
        # p = self.__compute_p([c1]) # only dataset size is considered (DS)
        # p = self.__compute_p([c2]) # only label diversity is considered (LD)
        # p = self.__compute_p([c3]) # only model divergence is considered (MD)
        # p = self.__compute_p([c1, c2, c3]) # DS, LD, MD
        # p = self.__compute_p([c1, c3, c2]) # DS, MD, LD
        # p = self.__compute_p([c2, c1, c3]) # LD, DS, MD
        # p = self.__compute_p([c3, c1, c2]) # MD, DS, LD
        # p = self.__compute_p([c2, c3, c1]) # LD, MD, DS
        # p = self.__compute_p([c3, c2, c1]) # MD, LD, DS

        # update server weights
        for layer in range(len(self.model.layers)):
            try:
                w = 0
                b = 0
                for k in range(m):
                    w += p[k] * clients[k].model.layers[layer].get_weights()[0]
                    b += p[k] * clients[k].model.layers[layer].get_weights()[1]
                self.model.layers[layer].set_weights([w, b])
            except:
                pass

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
        if self.iid:
            iid = "iid"
        else:
            iid = "non-iid"

        path = "experiments/" + iid + "/" + self.curr_time + "/" + "round_" + str(n_round) + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        if n_round == 1:
            with open("experiments/" + iid + "/" + self.curr_time + "/" + 'params.txt', 'w+') as file:
                file.write("Local dataset type: " + iid + "\r\n")
                file.write("Local learning rate: " + str(self.local_lr) + "\r\n")
                file.write("E (local epochs): " + str(self.E) + "\r\n")
                file.write("B (local batch size): " + str(self.B) + "\r\n")
                file.write("C (fraction of clients): " + str(self.C) + "\r\n")
                file.write("K (number of clients): " + str(self.K) + "\r\n")
                file.write("Target accuracy: " + str(target_acc) + "\r\n")
                file.write("Permutation: " + str(self.permutation))

            df = pd.DataFrame(data={'n_round': [n_round], '%_clients': [perc], 'aggr_accuracy': [accuracy]})
            df.to_csv("experiments/" + iid + "/" + self.curr_time + "/" + "accuracy.csv", index=False)
        else:
            with open("experiments/" + iid + "/" + self.curr_time + "/" + "accuracy.csv", 'a+', newline='') as write_obj:
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


