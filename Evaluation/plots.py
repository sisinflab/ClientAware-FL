import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def gen_labels_dict(dataset_name):
    test_labels_dict = {}
    if dataset_name == "MNIST":
        test_labels_dict[0] = "0_DS"
        test_labels_dict[1] = "1_LD"
        test_labels_dict[2] = "2_MD"
        test_labels_dict[3] = "3_DS_LD_MD"
        test_labels_dict[4] = "4_DS_MD_LD"
        test_labels_dict[5] = "5_LD_DS_MD"
        test_labels_dict[6] = "6_MD_DS_LD"
        test_labels_dict[7] = "7_LD_MD_DS"
        test_labels_dict[8] = "8_MD_LD_DS"
    elif dataset_name == "CelebA":
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

def gen_plots(path, out_path, dataset_name, target_acc):
    test_list = os.listdir(path)
    test_labels_dict = gen_labels_dict(dataset_name)

    for test_dir in test_list:
        curr_path = os.path.join(path,test_dir)
        aggr_acc = pd.read_csv(os.path.join(curr_path,"accuracy.csv"))["aggr_accuracy"].values

        aggr_acc = np.take(aggr_acc, np.arange(0, np.where(aggr_acc >= target_acc)[0][0] + 1))
        aggr_acc = np.insert(aggr_acc, 0, 0)

        plt.plot(aggr_acc)
        plt.xlabel('Communication rounds', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.locator_params(axis='x', nbins=4)

        x_ticks = plt.xticks()[0]
        x_ticks[-1] = aggr_acc.size-1
        plt.xticks(x_ticks)
        plt.tick_params(axis="x", labelsize = 15)
        plt.tick_params(axis="y", labelsize = 15)
        # plt.annotate(str(aggr_acc.size-1), xy=(aggr_acc.size-2, 0))

        with open(os.path.join(curr_path, "params.txt"), "r") as file:
            content = file.readlines()

        test_perm = int(content[-1].split(" ")[-1])

        x = np.linspace(0, aggr_acc.size)
        plt.plot(x, 0*x + target_acc, '--r')

        y = np.linspace(0, 1)
        plt.plot(0*y + aggr_acc.size-1, y, '--r')

        # plt.show()
        plt.savefig(out_path + "/" + test_labels_dict[test_perm] + ".jpg")
        plt.clf()

path = "CelebA"
out_path = "CelebA_plots"
dataset_name = "CelebA"
target_acc = 0.8
os.makedirs(out_path)
gen_plots(path, out_path, dataset_name, target_acc)

# def plot_aggr_accuracy(metrics):
#     plt.plot(metrics[:,1])
#     plt.plot(metrics[:,2])
#     plt.show()