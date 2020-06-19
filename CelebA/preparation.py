import pandas as pd
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import os
import pickle
import random

def CelebA(image_size, train_split = 0.8, test_split = 0.2):
    if train_split + test_split != 1:
        raise Exception("Splitting ratio not correct")

    path = "train_" + str(train_split).replace('.','') + "_test_" + str(test_split).replace('.','') + "_img_size_" + str(image_size)

    if os.path.exists(path):
        print("Finded a previous dataset version. Loading...")
        with open(path, 'rb') as file:
            return pickle.load(file)

    dataset = __CelebA(image_size)
    print("Splitting dataset in training and test set...")
    train_list = []
    test_list = []
    for client_ds in dataset:
        test_size = round(len(client_ds[0])*test_split)
        test_list.append([client_ds[0][:test_size], client_ds[1][:test_size], client_ds[2][:test_size]])
        train_list.append([client_ds[0][test_size:], client_ds[1][test_size:], client_ds[2][test_size:]])


    print("Saving training and test set...")
    with open(path,"wb") as file:
        pickle.dump((train_list, test_list), file)
    return train_list, test_list


def __CelebA(image_size):
    print("****GENERATING DATASET****")

    # source_path = "data/"
    source_path = "../../leaf/data/celeba/data/raw/"

    fin = open(source_path + "list_attr_celeba.txt", "rt")
    list_attr_celeba = fin.read()
    list_attr_celeba = list_attr_celeba[list_attr_celeba.find("\n")+1:]
    list_attr_celeba = 'Image ' + list_attr_celeba
    list_attr_celeba = list_attr_celeba.replace(' ', ',')
    list_attr_celeba = list_attr_celeba.replace(',,', ',')
    list_attr_celeba = list_attr_celeba.replace('.jpg', '.png')
    fin.close()

    fin = open(source_path + "identity_CelebA.txt", "rt")
    identity_CelebA = fin.read()
    identity_CelebA = identity_CelebA.replace(' ', ',')
    identity_CelebA = identity_CelebA.replace(',,', ',')
    identity_CelebA = identity_CelebA.replace('.jpg', '.png')
    fin.close()

    df = pd.read_csv(io.StringIO(list_attr_celeba))
    df = df.drop(df.columns[-1], axis=1)

    dfi = pd.read_csv(io.StringIO(identity_CelebA), header=None)
    dfi.columns = ['Image','Identity']
    df["Identity"] = dfi["Identity"].values

    dataset = []

    random.seed(0)
    by_id = [df for _, df in df.groupby('Identity')]
    random.shuffle(by_id)
    tot_id = int(len(by_id)/2) # half of the initial dataset shuffled
    by_id = by_id[:tot_id]

    for frame in by_id:
        if frame.shape[0] < 10:
            continue

        print("Generating dataset for client " + str(len(dataset) + 1))
        images_list = []
        blurry_list = []
        smiling_list = []
        for _, row in frame.iterrows():
            image = Image.open(source_path + "img_align_celeba/" + row["Image"])
            image = image.resize((image_size, image_size)).convert('RGB')
            image = np.array(image, dtype="float32") / 255
            if row["Blurry"] < 0:
                blurry = 0
            else:
                blurry = 1
            if row["Smiling"] < 0:
                smiling = 0
            else:
                smiling = 1
            images_list.append(image)
            blurry_list.append(blurry)
            smiling_list.append(smiling)


        dataset.append([images_list, blurry_list, smiling_list])
        # if len(dataset) == 50:
        #     break
    return dataset

# col = Image.open("cat-tied-icon.png")
# gray = col.convert('L')
# bw = gray.point(lambda x: 0 if x<128 else 255, '1')
# bw.save("result_bw.png")