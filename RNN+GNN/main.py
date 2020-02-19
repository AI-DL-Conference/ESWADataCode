import torch
import torch.nn.functional as F
import torch.optim as optim

from util import *
from args import get_args
from load_dataset import load_dataset
from model import *


def train(network, batch_embedding, label_train, lr, weight_decay, epoch_num):
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

    for iteration in range(epoch_num):
        network.train()
        optimizer.zero_grad()
        output = network(batch_embedding)
        loss_training = F.cross_entropy(output, label_train)
        loss_training.backward()
        optimizer.step()

    return network


def test(network, batch_embedding, label_test):
    network.eval()
    output = network(batch_embedding)

    accuracy = Accuracy(output, label_test)
    precision = Precision(output, label_test)
    recall = Recall(output, label_test)
    micro_f1, macro_f1 = F1(output, label_test)
    confusion = Confusion(output, label_test)

    return accuracy, precision, recall, micro_f1, macro_f1, confusion


def main():
    args = get_args()

    set_seed(args.seed, args.cuda)

    embedding_no, embedding_yes, label_no, label_yes = load_dataset()

    embedding_no_train = embedding_no[0:4900]
    label_no_train = label_no[0:4900]
    embedding_no_test = embedding_no[-1228:]
    label_no_test = label_no[-1228:]

    embedding_yes_train = embedding_yes[0:1878]
    label_yes_train = label_yes[0:1878]
    embedding_yes_test = embedding_yes[-470:]
    label_yes_test = label_yes[-470:]

    epoch_num = args.epochs

    network1 = Net(embedding_no.size(-1), args.hidden, args.num_layers).cuda()
    network2 = Net(embedding_yes.size(-1), args.hidden, args.num_layers).cuda()

    embedding_no_train = embedding_no_train.transpose(1, 0)
    network1 = train(network1, embedding_no_train, label_no_train, args.lr, args.weight_decay, epoch_num)
    test_accuracy, test_precision, test_recall, test_micro_f1, test_macro_f1, confusion = test(network1, embedding_no_test.transpose(1, 0), label_no_test)
    print('(0) Testing Accuracy: {} Testing Precision: {} Testing Recall: {} Testing Micro F1: {} Testing Macro F1: {}'.format(test_accuracy, test_precision, test_recall, test_micro_f1, test_macro_f1))
    print(confusion)

    embedding_yes_train = embedding_yes_train.transpose(1, 0)
    network2 = train(network2, embedding_yes_train, label_yes_train, args.lr, args.weight_decay, 2500)
    test_accuracy, test_precision, test_recall, test_micro_f1, test_macro_f1, confusion = test(network2, embedding_yes_test.transpose(1, 0), label_yes_test)
    print('(1) Testing Accuracy: {} Testing Precision: {} Testing Recall: {} Testing Micro F1: {} Testing Macro F1: {}'.format(test_accuracy, test_precision, test_recall, test_micro_f1, test_macro_f1))
    print(confusion)


if __name__ == '__main__':
    main()
