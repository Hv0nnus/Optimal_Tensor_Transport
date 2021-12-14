# import torch
import argparse
# import torchvision
from COT import cot_numpy,random_gamma_init

# import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
import gromov
import ot
import pickle
# import matplotlib.ticker
import ot.plot
import time
from copy import deepcopy

import struct



def loadlocal_mnist(images_path, labels_path):
    """ Read MNIST from ubyte files.
    Parameters
    ----------
    images_path : str
        path to the test or train MNIST ubyte file
    labels_path : str
        path to the test or train MNIST class labels file
    Returns
    --------
    images : [n_samples, n_pixels] numpy.array
        Pixel values of the images.
    labels : [n_samples] numpy array
        Target class labels
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
    """
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def compute_save_T(labels_selected=[[0, 1], [0, 1]],
                   n_per_features=[200, 200],
                   time_max=1,
                   seed=42,
                   coot=False,
                   save=False):
    np.random.seed(12345)
    C = [None, None]
    labels = [None, None]
    path_root = './data'
    for c in range(2):
        if c == 0:
            path = "./data/MNIST/raw/"
            X, y = loadlocal_mnist(images_path=path + 'train-images-idx3-ubyte',
                                   labels_path=path + 'train-labels-idx1-ubyte')  # download the MNIST dataset
            X = X.reshape(X.shape[0], 28, 28)
        else:
            # As there is too much problem with USPS loading with torchvision and local, I create a pickle.
            if False:
                trainset = datasets.USPS(root=path_root, train=True, download=False, transform=None)
                X, y = trainset.data, trainset.targets
                with open("./data/pickle_usps/usps.pickle", "wb") as f:
                    pickle.dump({"X": X, "y": y}, f)
            else:
                with open("./data/pickle_usps/usps.pickle", "rb") as f:  # put True above for the first time
                    dict_usps = pickle.load(f)
                X, y = dict_usps["X"], dict_usps["y"]
        labels_init = np.array(y)
        features_init = np.array(X)
        for i, label_selected in enumerate(labels_selected[c]):
            selections = np.isin(labels_init, label_selected)
            labels_, features_ = labels_init[selections], features_init[selections]
            selections_rdm = np.random.choice(len(labels_), size=n_per_features[c])
            labels_, features_ = labels_[selections_rdm], features_[selections_rdm]
            if C[c] is None:
                C[c] = features_
                labels[c] = labels_
            else:
                C[c] = np.concatenate((C[c], features_), axis=0)
                labels[c] = np.concatenate((labels[c], labels_), axis=0)

    C1, C2 = C[0]/C[0].mean(), C[1]/C[1].mean()
    # print(C1.sum(), C2.sum())

    p = [ot.unif(C1.shape[i]) for i in range(3)]
    q = [ot.unif(C2.shape[i]) for i in range(3)]

    def loss_fun(C1, C2):
        return (C1 - C2) ** 2

    def X1(*x):
        return C1[x]

    def X2(*x):
        return C2[x]
    X1_COOT = C1.reshape((np.sum(n_per_features)), -1)

    X2_COOT = C2.reshape((np.sum(n_per_features)), -1)

    #     print(labels)
    #     print(p[1].shape, q[1].shape)

    T_init = [None, None, None]
    time_init = time.time()
    best_gw_dist = np.inf
    i = 0
    log_ = {}
    while (time.time() - time_init) < time_max * 3600:
        print(i)
        np.random.seed(seed * i + seed)
        if coot:
            Ts, Tv, ot_dist = cot_numpy(X1=X1_COOT, X2=X2_COOT, w1=None, w2=None, v1=None, v2=None,
                                        labels_s=[labels[0], None],
                                        labels_t=[None, None],
                                        niter=10, algo='supervised', reg=0, algo2='supervised',
                                        reg2=0,
                                        eta=1, verbose=False, log=False, random_init=True, C_lin=None)
            log_["gw_dist_estimated"] = ot_dist
            T = [Ts, Tv]
        else:
            T, log_ = gromov.CO_Generalisation_OT(p=p, q=q,
                                                  loss_fun=[loss_fun],
                                                  X1=[X1], X2=[X2],
                                                  M=[None],
                                                  T_pos=[[0, 1, 2]],
                                                  T=T_init,
                                                  alpha=[1],
                                                  nb_iter=1000,
                                                  nb_samples=[1000],
                                                  nb_samples_t=None,
                                                  epsilon=0.1,  # 1
                                                  KL=1,
                                                  L2=0,
                                                  labels_t=None,
                                                  labels_s=[labels[0], None, None],
                                                  eta=0.1,  # 1
                                                  log=True,
                                                  sliced=False,
                                                  learning_step=1,
                                                  verbose=args["verbose"],
                                                  sparse_T=True,
                                                  threshold=1e-20,
                                                  time_print=False,
                                                  sample_t_only_init=False,
                                                  sample_t_init_and_iteration=False)
        print("log_[gw_dist_estimated]", log_["gw_dist_estimated"])
        print("best_gw_dist", best_gw_dist)
        print("")
        if log_["gw_dist_estimated"] < best_gw_dist:
            best_gw_dist = log_["gw_dist_estimated"]
            log = deepcopy(log_)
        i = i + 1

    log["T"] = T
    log["C1"], log["C2"] = C1, C2
    log["labels1"], log["labels2"] = labels[0], labels[1]
    labels_selected_str = ""
    for labels_str in labels_selected[0]:
        labels_selected_str = labels_selected_str + "," + str(labels_str)
    print('./visualization/pickle_save/visu_COOT_' + labels_selected_str + '.pickle')
    if save:
        if coot:
            with open('./visualization/pickle_save/visu_COOT_' + labels_selected_str + '.pickle', 'wb') as handle:
                pickle.dump(log, handle)
        else:
            with open('./visualization/pickle_save/visu' + labels_selected_str + '.pickle', 'wb') as handle:
                pickle.dump(log, handle)


def plot_images(image1, image2, only_T=False, save=False, init_id=False,
                labels_selected_str="0,1,2",
                path1='./visualization/pickle_save/', path2='./visualization/pickle_save/'):

    with open(path1 + 'visu' + labels_selected_str + '.pickle', 'rb') as handle:
        dict_ = pickle.load(handle)

    T = dict_["T"]
    n = 1
    for i in range(len(T[0])):
        pos_T_i = T[0][i].argsort()[-n:][::-1]
        plt.plot(i * np.ones(n), pos_T_i, ".b", markersize=8)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([100, 200, 300])
    ax.set_yticks([100, 200, 300])
    yticks = ax.yaxis.get_major_ticks()
    yticks[1].tick1line.set_visible(False)
    xticks = ax.xaxis.get_major_ticks()
    xticks[1].tick1line.set_visible(False)
    ax.set_xticklabels([0, "MNIST", 1])
    ax.set_yticklabels([0, "USPS", 1])
    ax.tick_params(axis='both', labelsize=15)
    if save:
        plt.savefig("./visualization/confusion.eps", bbox_inches='tight')
    plt.show()

    plt.imshow(T[1], cmap="Greys", interpolation='nearest')
    if save:
        plt.savefig("./visualization/line.eps", bbox_inches='tight')
    plt.show()

    plt.imshow(T[2].T, cmap="Greys", interpolation='nearest')
    if save:
        plt.savefig("./visualization/column.eps", bbox_inches='tight')
    plt.show()
    if only_T:
        return None

    im = plt.imshow(dict_["C1"][image1], cmap='Greys', interpolation='nearest')
    ax = plt.gca()
    ax.set_xticks([-5])
    ax.set_yticks([-5])
    #         np.arange(0, dict_["C1"].shape[2], 1))
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    if save:
        plt.savefig("./visualization/MNIST.eps", bbox_inches='tight')

    plt.show()
    im = plt.imshow(dict_["C2"][image2], cmap='Greys', interpolation='nearest', aspect='equal')
    ax = plt.gca()
    ax.set_xticks([-5])
    ax.set_yticks([-5])
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    if save:
        plt.savefig("./visualization/USPS.eps", bbox_inches='tight')
    plt.show()

    n, m = T[1].shape[0], T[2].shape[1]

    fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [n, m], "height_ratios": [n, m]}, figsize=(4.5, 4.5))
    fig.subplots_adjust(hspace=0.05, wspace=0.01)
    axs[0, 0].imshow(dict_["C1"][image1], cmap='Greys', interpolation='nearest', aspect='equal')
    # axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0, 0].tick_params(axis='both', which='both', bottom=False, top=False,  left=False, right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    axs[0, 1].imshow(T[1], cmap="Greys", interpolation='nearest')
    axs[0, 1].tick_params(axis='both', which='both', bottom=False, top=True, left=False, right=True,
                          labelbottom=False, labeltop=True, labelleft=False, labelright=True)
    axs[1, 0].imshow(T[2].T, cmap="Greys", interpolation='nearest')
    axs[1, 0].tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False,
                          labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    axs[1, 1].imshow(dict_["C2"][image2].T, cmap='Greys', interpolation='nearest', aspect='equal')
    axs[1, 1].tick_params(axis='both', which='both', bottom=False, top=False,  left=False, right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    if save:
        plt.savefig("./visualization/grid.eps", bbox_inches='tight')
    plt.show()


    dim_source = 16
    dim_target = 28
    Tl, Tc = T[1].T, T[2].T

    to_keep_l_1 = np.sum(dict_["C1"], axis=(0, 1)) != 0
    to_keep_c_1 = np.sum(dict_["C1"], axis=(0, 2)) != 0

    Tl, Tc = Tl[:, to_keep_l_1], Tc[:, to_keep_c_1]
    dim_target_l, dim_target_c = np.sum(to_keep_l_1), np.sum(to_keep_c_1)

    plt.imshow(Tl, cmap="Greys", interpolation='nearest')
    #     plt.title('OT matrix G0')
    if save:
        plt.savefig("./visualization/line.eps", bbox_inches='tight')
    plt.show()

    plt.imshow(Tc.T, cmap="Greys", interpolation='nearest')

    if save:
        plt.savefig("./visualization/column.eps", bbox_inches='tight')
    plt.show()

    image = np.zeros((dim_source, dim_source, 3))

    for i in range(dim_source):
        for j in range(dim_source):
            image[i, j, 0] = i
            image[i, j, 1] = j
            image[i, j, 2] = dim_source / 2
    image = image.astype(np.float32) / dim_source
    plt.imshow(image)
    plt.title('source image')
    plt.axis('off')
    plt.show()

    image_target = np.zeros((dim_target_l, dim_target_c, 3))
    for i in range(dim_source):
        for j in range(dim_source):
            a = image[i, j, np.newaxis, np.newaxis, :] * Tl[i, :, np.newaxis, np.newaxis] * Tc[np.newaxis, j, :,  np.newaxis]
            image_target += a

    for c in range(3):
        if image_target[:, :, c].mean() > 0:
            image_target[:, :, c] = image[:, :, c].mean() * image_target[:, :, c] / image_target[:, :, c].mean()

    plt.imshow(image_target)
    plt.axis('off')
    if save:
        plt.savefig("./visualization/OTT.eps", bbox_inches='tight')
    plt.show()


    #COOT
    with open(path2 + 'visu_COOT_' + labels_selected_str + '.pickle', 'rb') as handle:
        dict_ = pickle.load(handle)
    Tv = Tvr = dict_["T"][1]
    dim_source = 16
    dim_target = 28

    image = np.zeros((dim_source, dim_source, 3))

    for i in range(dim_source):
        for j in range(dim_source):
            image[i, j, 0] = i
            image[i, j, 1] = j
            image[i, j, 2] = dim_source / 2
    image = image.astype(np.float32) / dim_source

    diag = 1. / Tv.sum(axis=1)
    diag[diag == np.inf] = 0
    image_target = np.dot(np.diag(diag), np.dot(image.reshape((dim_source * dim_source, 3)).T, Tv.T).T)

    # image_target[~selmnist, :] = np.nan  # we remove non informative features

    image_target = image_target.reshape((dim_target, dim_target, 3))

    diagr = 1. / Tvr.sum(axis=1)
    diagr[diagr == np.inf] = 0
    image_targetr = np.dot(np.diag(diagr), np.dot(image.reshape((dim_source * dim_source, 3)).T, Tvr.T).T)

    # image_targetr[~selmnist, :] = np.nan

    image_targetr = image_targetr.reshape((dim_target, dim_target, 3))


    plt.imshow(image)
    plt.axis('off')
    if save:
        plt.savefig("./visualization/init.eps", bbox_inches='tight')
    plt.title('source image')
    plt.show()
    plt.imshow(image_target)
    plt.axis('off')
    if save:
        plt.savefig("./visualization/COOT.eps", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OTT')
    parser.add_argument('--labels_selected', type=str, default="0,1")
    parser.add_argument('--n_per_features', type=str, default="200,200")
    parser.add_argument("-c", "--coot", action="store_true")
    parser.add_argument('--time_max', type=float, default=24.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Will print more information about the run")
    args = parser.parse_args()
    args = vars(args)
    # labels_selected = [[0, 1], [0, 1]]
    labels_selected = [int(x) for x in args["labels_selected"].split(",")]
    labels_selected = [labels_selected, labels_selected]
    n_per_features = [int(x) for x in args["n_per_features"].split(",")]
    time_max = args["time_max"]
    seed = args["seed"]
    compute_save_T(labels_selected=labels_selected,
                   n_per_features=n_per_features,
                   time_max=time_max,
                   seed=seed,
                   coot=args["coot"],
                   save=args["save"])
