import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse

def get_metrics(path, target_acc):
    round_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    metrics = []
    for round_dir in round_list:
        n_round = int(round_dir.split(sep='_')[1])
        curr_path = os.path.join(path, round_dir)
        print("Processing " + curr_path)

        acc_list = []
        client_list = os.listdir(curr_path)
        for client_file in client_list:
            y_client = pd.read_csv(os.path.join(curr_path, client_file)).values
            y_pred = y_client[:, 0]
            y_true = y_client[:, 1]
            acc_list.append(accuracy_score(y_true, y_pred))

        perc = evaluate(acc_list, target_acc)
        metrics.append((n_round, perc))

    metrics = np.array(metrics)
    metrics = metrics[np.argsort(metrics[:, 0])]
    return metrics

def gen_table_row(metrics, perc_list):
    round_list = []
    for perc in perc_list:
        index = np.argmax(metrics[:, 1] >= perc)
        if metrics[index, 1] >= perc:
            n_round = int(metrics[index, 0])
        else:
            n_round = np.NAN
        round_list.append(n_round)

    return round_list

def evaluate(acc_list, target_acc):

    n_sat_clients = np.where(np.array(acc_list) > target_acc)[0].size
    perc = (n_sat_clients*100)/len(acc_list)

    return perc

def gen_table_csv(path, perc_list, target_acc, test_labels_dict):
    table_filename = path.split("/")[-1] + str(int(target_acc*100)) + "_table.csv"
    test_list = os.listdir(path)
    test_labels = []
    table = []
    for test in test_list:
        test_path = os.path.join(path, test)
        metrics = get_metrics(test_path, target_acc)
        table.append(gen_table_row(metrics, perc_list))

        with open(os.path.join(test_path, "params.txt"), "r") as file:
            content = file.readlines()

        test_labels.append(int(content[-1].split(" ")[-1]))

    if test_labels_dict is not None:
        test_labels = [test_labels_dict[value] for value in test_labels]

    table = np.array(table)



    df = pd.DataFrame(columns=perc_list, index=test_labels, data=table)
    df = df.sort_index()
    df.to_csv(table_filename)

def gen_labels_dict(dataset_list, dataset_name):
    test_labels_dict = None
    if dataset_name is not None:
        test_labels_dict = {}
        if dataset_name in dataset_list:
            if args.dataset == dataset_list[0]:  # MNIST
                test_labels_dict[0] = "0_DS"
                test_labels_dict[1] = "1_LD"
                test_labels_dict[2] = "2_MD"
                test_labels_dict[3] = "3_DS_LD_MD"
                test_labels_dict[4] = "4_DS_MD_LD"
                test_labels_dict[5] = "5_LD_DS_MD"
                test_labels_dict[6] = "6_MD_DS_LD"
                test_labels_dict[7] = "7_LD_MD_DS"
                test_labels_dict[8] = "8_MD_LD_DS"
            elif args.dataset == dataset_list[1]:  # CelebA
                test_labels_dict[0] = "0_DS"
                test_labels_dict[1] = "1_IS"
                test_labels_dict[2] = "2_CB"
                test_labels_dict[3] = "3_DS_IS_CB"
                test_labels_dict[4] = "4_DS_CB_IS"
                test_labels_dict[5] = "5_IS_DS_CB"
                test_labels_dict[6] = "6_CB_DS_IS"
                test_labels_dict[7] = "7_IS_CB_DS"
                test_labels_dict[8] = "8_CB_IS_DS"

        else:
            raise Exception("The specified dataset is not supported")
    return test_labels_dict

# def plot_aggr_accuracy(metrics):
#     plt.plot(metrics[:,1])
#     plt.plot(metrics[:,2])
#     plt.show()


dataset_list = ["MNIST", "CelebA"]
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path",
                    help="Directory path in which are placed experiments. E.g MNIST/iid",
                    type=str)
parser.add_argument("-ta", "--target-accuracy",
                    help="Target accuracy to consider for stats generation. Real number in range [0,1]. E.g. 0.95",
                    type=float)

parser.add_argument("-pl", "--percentage-list", nargs='+',
                    help="List of client percentages that exceed the target accuracy choosen. E.g. 50 60 80 90",
                    type=int)
parser.add_argument("-ds", "--dataset",
                    help="Possible values: " + ', '.join(dataset_list),
                    type=str,
                    default=None)

args = parser.parse_args()

test_labels_dict = gen_labels_dict(dataset_list, args.dataset)
gen_table_csv(args.path, args.percentage_list, args.target_accuracy, test_labels_dict)

# gen_table_csv("MNIST/iid", [50, 70, 90, 95], 0.95)






# path = "../1_034815_23Mar2020/"
# target_acc = 0.95
# K = 100
#
# rounds_stats = {}
#
# for n_round in range(1, 501 + 20, 20):
#     n_sat_clients = 0
#     for i in range(100):
#         curr_path = path + "round_" + str(n_round) + "/client_" + str(i) + ".csv"
#         y_client = pd.read_csv(curr_path).values
#         y_pred = y_client[:, 0]
#         y_true = y_client[:, 1]
#
#         # cr = classification_report(y_true, y_pred, output_dict=True)
#         # cm = confusion_matrix(y_true, y_pred)
#         if accuracy_score(y_true, y_pred) >= target_acc:
#             n_sat_clients += 1
#
#     perc = (n_sat_clients*100)/K
#     rounds_stats[n_round] = perc
#
#
# df = pd.DataFrame(data={'n_round': list(rounds_stats.keys()), '%_clients': list(rounds_stats.values())})
# print(df)






