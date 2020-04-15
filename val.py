#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Group Normalization" Implementation
20193640 Jungwon Choi
'''
import numpy as np
import sys
#===============================================================================
''' Validate sequence '''
def val(model, val_loader):
    # Check val error
    model.eval()
    device = next(model.parameters()).device.index
    pred_labels_list = []
    real_labels_list = []
    total_iter = len(val_loader)

    for i, (images, labels) in enumerate(val_loader):
        images, labels = images.cuda(device), labels.cuda(device)
        # labels = labels.view(labels.size(0))

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
    return acc, err
