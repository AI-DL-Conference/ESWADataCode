import torch
import numpy as np

from sklearn.metrics import *


def Accuracy(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return accuracy_score(labels, preds)


def F1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro


def Recall(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return recall_score(labels, preds, average='micro')


def Precision(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return precision_score(labels, preds, average='micro')


def Confusion(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return confusion_matrix(labels, preds)


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
