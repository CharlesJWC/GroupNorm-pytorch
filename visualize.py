#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Group Normalization" Implementation
20193640 Jungwon Choi
'''
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.rc('font',family='Cambria')
# mpl.rc('font',family='Times New Roman')
SAVE_PATH = './figures'
DATA_PATH = './results'

#===============================================================================
''' Experiment 1-1 results (train & val error graph) '''
def visualize_ex1_1_results(batch_size=32, model_depth=56):
    scale_lr = 0.1*batch_size/32
    file_list = [
        "resnet{:d}_100_{:d}_BN_None_{:.4f}.pkl".format(model_depth, batch_size, scale_lr),
        "resnet{:d}_100_{:d}_LN_None_{:.4f}.pkl".format(model_depth, batch_size, scale_lr),
        "resnet{:d}_100_{:d}_IN_None_{:.4f}.pkl".format(model_depth, batch_size, scale_lr),
        "resnet{:d}_100_{:d}_GN_16_{:.4f}.pkl".format(model_depth, batch_size, scale_lr),
        ]
    labels = [
        'Batch Norm (BN)',
        'Layer Norm (LN)',
        'Instance Norm (IN)',
        'Group Norm (GN)',
        ]

    # Load trina & val err data
    train_err_results = []
    val_err_results = []
    for FILE_PATH in file_list:
        with open(os.path.join(DATA_PATH,FILE_PATH), 'rb') as pkl_file:
            results = pickle.load(pkl_file)
        train_err_results.append(results['train_err'])
        val_err_results.append(results['val_err'])

    # Plot train error =========================================================
    num_epoch = len(train_err_results[0])
    epochs = np.arange(1, num_epoch+1)
    fig = plt.figure(dpi=150)
    plt.title('Train error')
    plt.xlabel('epochs')
    plt.ylabel('error(%)')
    plt.xlim([0, num_epoch])
    plt.ylim([0, 60])
    for train_error, label in zip(train_err_results, labels):
        plt.plot(epochs, train_error,'-.', markersize=1, alpha=0.8, label=label)
    plt.legend()

    # Save figure
    file_name = "Experiment1_train_error_ResNet{:d}_{:d}.png".format(model_depth, batch_size)
    fig.savefig(os.path.join(SAVE_PATH, file_name),format='png')

    # Plot val error ===========================================================
    num_epoch = len(val_err_results[0])
    epochs = np.arange(1, num_epoch+1)
    fig = plt.figure(dpi=150)
    plt.title('Val error')
    plt.xlabel('epochs')
    plt.ylabel('error(%)')
    plt.xlim([0, num_epoch])
    plt.ylim([0, 60])
    for val_error, label in zip(val_err_results, labels):
        plt.plot(epochs, val_error,'--', markersize=1, alpha=0.8, label=label)
    plt.legend()

    file_name = "Experiment1_val_error_ResNet{:d}_{:d}.png".format(model_depth, batch_size)
    fig.savefig(os.path.join(SAVE_PATH, file_name),format='png')


''' Experiment 1-2 results (val error graph per batch) '''
def visualize_ex1_2_results(batch_size_list, model_depth=56):
    labels = [
        'Batch Norm (BN)',
        'Layer Norm (LN)',
        'Instance Norm (IN)',
        'Group Norm (GN)',
        ]

    val_err_BN_epochs = []
    val_err_LN_epochs = []
    val_err_IN_epochs = []
    val_err_GN_epochs = []

    val_err_norm_list = [val_err_BN_epochs, val_err_LN_epochs, val_err_IN_epochs, val_err_GN_epochs]

    for batch_size in batch_size_list:
        scale_lr = 0.1*batch_size/32
        file_list = [
            "resnet{:d}_100_{:d}_BN_None_{:.4f}.pkl".format(model_depth, batch_size, scale_lr),
            "resnet{:d}_100_{:d}_LN_None_{:.4f}.pkl".format(model_depth, batch_size, scale_lr),
            "resnet{:d}_100_{:d}_IN_None_{:.4f}.pkl".format(model_depth, batch_size, scale_lr),
            "resnet{:d}_100_{:d}_GN_16_{:.4f}.pkl".format(model_depth, batch_size, scale_lr),
            ]
        for FILE_PATH, val_err_epochs in zip(file_list, val_err_norm_list):
            with open(os.path.join(DATA_PATH,FILE_PATH), 'rb') as pkl_file:
                results = pickle.load(pkl_file)
            val_err_epochs.append(results['val_err'])

    norm_list = ['BN',
                'LN',
                'IN',
                'GN']

    # Plot val error ===========================================================
    for val_err_results, norm_type in zip(val_err_norm_list, norm_list):
        num_epoch = len(val_err_results[0])
        epochs = np.arange(1, num_epoch+1)
        fig = plt.figure(dpi=150)
        plt.title('Val error')
        plt.xlabel('epochs')
        plt.ylabel('error(%)')
        plt.xlim([0, num_epoch])
        plt.ylim([0, 60])
        for val_error, batch_size in zip(val_err_results, batch_size_list):
            plt.plot(epochs, val_error,'-', markersize=1, alpha=1, label='{}, {:d} img/gpu'.format(norm_type,batch_size))
        plt.legend()

        file_name = "Experiment1_2_val_error_ResNet{:d}_{}.png".format(model_depth, norm_type)
        fig.savefig(os.path.join(SAVE_PATH, file_name),format='png')


    val_err_BN = []
    val_err_LN = []
    val_err_IN = []
    val_err_GN = []

    val_err_list = [val_err_BN,
                    val_err_LN,
                    val_err_IN,
                    val_err_GN]

    for val_err_epochs_list, val_err in zip(val_err_norm_list, val_err_list):
        for val_err_epochs in val_err_epochs_list:
            val_err.append(np.median(val_err_epochs[-5:]))

    # reverse data: big -> small
    # for err_list in val_err_list:
    #     err_list.reverse()
    #     # print(err_list)
    # batch_size_list.reverse()

    # Plot val error ===========================================================
    fig = plt.figure(dpi=150)
    x = np.arange(len(batch_size_list))
    plt.title('Val error')
    plt.xlabel('batch size')
    plt.ylabel('error(%)')
    plt.xlim([0, len(x)-1])
    plt.ylim([6,15])
    marker_list = ['+',
                   '^',
                   'd',
                   'o']
    for val_error, label, marker in zip(val_err_list, labels, marker_list):
        plt.plot(x, val_error,'-'+marker, markersize=7, alpha=1, label=label)
    plt.legend()
    plt.xticks(x, batch_size_list)

    file_name = "Experiment1_2_val_error_ResNet{:d}_All.png".format(model_depth)
    fig.savefig(os.path.join(SAVE_PATH, file_name),format='png')

#===============================================================================
''' Experiment 2 results '''
def visualize_ex2_results():
    file_list = [
        "resnet56_100_32_GN_1_0.1000.pkl",
        "resnet56_100_32_GN_2_0.1000.pkl",
        "resnet56_100_32_GN_4_0.1000.pkl",
        "resnet56_100_32_GN_8_0.1000.pkl",
        "resnet56_100_32_GN_16_0.1000.pkl",
        ]
    labels = [
        '1 group',
        '2 group',
        '4 group',
        '8 group',
        '16 group',
        ]

    # Load trina & val err data
    train_err_results = []
    val_err_results = []
    for FILE_PATH in file_list:
        with open(os.path.join(DATA_PATH,FILE_PATH), 'rb') as pkl_file:
            results = pickle.load(pkl_file)
        train_err_results.append(results['train_err'])
        val_err_results.append(results['val_err'])

    # Plot train error =========================================================
    num_epoch = len(train_err_results[0])
    epochs = np.arange(1, num_epoch+1)
    fig = plt.figure(dpi=150)
    plt.title('Train error')
    plt.xlabel('epochs')
    plt.ylabel('error(%)')
    plt.xlim([0, num_epoch])
    plt.ylim([0, 60])
    for train_error, label in zip(train_err_results, labels):
        plt.plot(epochs, train_error,'-.', markersize=1, alpha=0.8, label=label)
    plt.legend()

    # Save figure
    file_name = "Experiment2_train_error.png"
    fig.savefig(os.path.join(SAVE_PATH, file_name),format='png')

    # Plot val error ===========================================================
    num_epoch = len(val_err_results[0])
    epochs = np.arange(1, num_epoch+1)
    fig = plt.figure(dpi=150)
    plt.title('Val error')
    plt.xlabel('epochs')
    plt.ylabel('error(%)')
    plt.xlim([0, num_epoch])
    plt.ylim([0, 60])
    for val_error, label in zip(val_err_results, labels):
        plt.plot(epochs, val_error,'--', markersize=1, alpha=0.8, label=label)
    plt.legend()

    file_name = "Experiment2_val_error.png"
    fig.savefig(os.path.join(SAVE_PATH, file_name),format='png')

    # Print median error =======================================================
    train_err_median_list = []
    val_err_median_list = []
    for train_err, val_err in zip(train_err_results, val_err_results):
        train_err_median_list.append(np.median(train_err[-5:]))
        val_err_median_list.append(np.median(val_err[-5:]))
    labels = ['1','2','4','8','16']
    labels.reverse()
    print('\tExprement2 results')
    print('\t************ # Groups (G) ************')
    val_err_median_list.reverse()
    for num_group in labels:
        sys.stdout.write('\t'+num_group)
    print()
    for val_err_median in val_err_median_list:
        sys.stdout.write('\t{:.1f}'.format(val_err_median))
    print()
    std = val_err_median_list[0]
    for i, val_err_median in enumerate(val_err_median_list):
        if i == 0:
            sys.stdout.write('\t')
            continue
        sys.stdout.write('\t{:.1f}'.format(val_err_median-std))
    print()

#===============================================================================
''' Experiment Results plotting '''
if __name__ == '__main__':

    visualize_ex1_1_results(8)
    # visualize_ex1_1_results(16)
    visualize_ex1_1_results(32)
    # visualize_ex1_1_results(64)
    # visualize_ex1_1_results(128)
    # visualize_ex1_1_results(256)
    # visualize_ex1_1_results(512)

    visualize_ex1_2_results([256,128,64,32,16,8], 56)
    # visualize_ex1_2_results([256,128,64,32,16], 110)
    visualize_ex2_results()
