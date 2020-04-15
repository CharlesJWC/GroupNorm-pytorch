#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Group Normalization" Implementation
20193640 Jungwon Choi
'''
import numpy as np
import sys

#===============================================================================
''' Train sequence '''
def train(model, train_loader, criterion, optimizer):
    model.train()
    device = next(model.parameters()).device.index
    losses = []
    total_iter = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        # images = data[0].type(torch.FloatTensor).cuda(device)
        # labels = data[1].type(torch.LongTensor).cuda(device)
        images, labels = images.cuda(device), labels.cuda(device)
        # labels = labels.view(labels.size(0))

        # Predict labels (Forward propagation)
        pred_labels = model(images)

        # Calculate loss
        loss = criterion(pred_labels, labels)
        losses.append(loss.item())

        # Empty gradients
        optimizer.zero_grad()

        # Calculate gradients (Backpropagation)
        loss.backward()

        # Update parameters
        optimizer.step()

        sys.stdout.write("[{:5d}/{:5d}]\r".format(i+1, total_iter))

    avg_loss = sum(losses)/len(losses)

    #===========================================================================
    # Check train error
    model.eval()
    pred_labels_list = []
    real_labels_list = []

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(device), labels.cuda(device)

        # Predict labels (Forward propagation)
        pred_labels = model(images)

        # Accumulate the results
        real_labels_list += list(labels.cpu().detach().numpy())
        pred_labels_list += list(pred_labels.cpu().detach().numpy())

        sys.stdout.write("[{:5d}/{:5d}]\r".format(i+1, total_iter))

    # Calculate accuracy
    real_labels_np = np.array(real_labels_list)
    pred_labels_np = np.array(pred_labels_list).argmax(axis=1)
    acc = sum(real_labels_np==pred_labels_np)/len(real_labels_np)*100
    err = 100-acc
    return avg_loss, acc, err
