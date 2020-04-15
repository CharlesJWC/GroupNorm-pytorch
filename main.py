#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Group Normalization" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import pickle
import time

# Implementation files
from dataloader import CIFAR10_Dataloader
from model.ResNet import ResNet56, ResNet110
from train import train
from val import val

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

#===============================================================================
''' Main Function '''
def main(args):
    # Step1 ====================================================================
    # Load dataset
    cifar10 = CIFAR10_Dataloader()
    train_loader = cifar10.get_train_loader(batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    val_loader = cifar10.get_val_loader(batch_size=args.batch_size, # 10000개?
                                            num_workers=args.num_workers)
    print('==> DataLoader ready.')

    # Step2 ====================================================================
    # Make training model (ResNet56, ResNet110 for CIFAR-10)
    if args.model == 'resnet56':
        model = ResNet56(args.norm_type, args.num_groups)
    elif args.model == 'resnet110':
        model = ResNet110(args.norm_type, args.num_groups)
    else:
        assert False, "Select model"

    # Check DataParallel available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Check CUDA available
    if torch.cuda.is_available():
        model.cuda()
    print('==> Model ready.')

    # Step3 ====================================================================
    # Set loss function and optimizer

    # Applying linear scaling rule for learning rate
    N = args.batch_size
    scale_lr = args.lr*N/32

    # Parameter reference [17]
    # https://github.com/facebook/fb.resnet.torch/blob/master/opts.lua
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=scale_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # Learning rate scheduler : decrease lr by X10 per 30 epochs
    print('==> Training ready.')

    # Step4 ====================================================================
    # Train and validate the model

    train_loss = []
    train_acc = []
    train_err = []
    val_acc = []
    val_err = []

    FILE_PATH = "./results/{0}_{1}_{2}_{3}_{4}_{5:.4f}.csv".format(args.model,
                        str(args.epochs), str(args.batch_size), args.norm_type,
                        str(args.num_groups), scale_lr)

    for epoch in range(args.epochs):
        print("\n[Epoch: {:3d}/{:3d}]".format(epoch+1, args.epochs))
        csv_file = open(FILE_PATH,'a')
        epoch_time = time.time()
        #=======================================================================
        # train the model
        loss, tacc, terr = train(model, train_loader, criterion, optimizer)
        train_loss.append(loss)
        train_acc.append(tacc)
        train_err.append(terr)

        # validate the model
        vacc, verr = val(model, val_loader)
        val_acc.append(vacc)
        val_err.append(verr)

        # epoch information update
        scheduler.step()
        #=======================================================================
        current = time.time()
        csv_file.write("{},{},{}\n".format(epoch+1, terr, verr))
        print("loss           : {:.3f}".format(loss))
        print("lr             : {:.4f}".format(optimizer.param_groups[0]['lr']))
        print("train acc/err  : {:.1f}/{:.1f}".format(tacc, terr))
        print("val   acc/err  : {:.1f}/{:.1f}".format(vacc, verr))
        print("Epoch time     : {0:.3f} sec".format(current - epoch_time))
        print("Current elapsed time: {0:.3f} sec\n".format(current - start))
        csv_file.close()
    print('==> Train Done.')
    # Step5 ====================================================================
    # Save the train and val information
    result_data = {}
    result_data['model']        = args.model
    result_data['batch_size']   = args.batch_size
    result_data['lr']           = scale_lr
    result_data['epochs']       = args.epochs
    result_data['norm_type']    = args.norm_type
    result_data['num_groups']   = args.num_groups
    result_data['train_loss']   = train_loss
    result_data['train_acc']    = train_acc
    result_data['train_err']    = train_err
    result_data['val_acc']      = val_acc
    result_data['val_err']      = val_err

    FILE_PATH = "./results/{0}_{1}_{2}_{3}_{4}_{5:.4f}.pkl".format(args.model,
                        str(args.epochs), str(args.batch_size), args.norm_type,
                        str(args.num_groups), scale_lr)

    # Save result_data as pkl file
    with open(FILE_PATH, 'wb') as pkl_file:
        pickle.dump(result_data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(' '.join(['[Done] Results have been saved at', FILE_PATH]))

#===============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GroupNorm Implementation using CIFAR-10')
    parser.add_argument('--model', default=None, type=str, help='resnet56, resnet110')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--norm_type', default='GN', type=str, help='BN, GN, LN, IN')
    parser.add_argument('--num_groups', default=None, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()

    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("Total elapsed time: {0:.3f} sec\n".format(end - start))
