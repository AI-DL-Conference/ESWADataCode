import numpy as np
import torch


def load_dataset(cuda=True):
    print('Loading Data')
    embedding_no = np.load('./no.npy')
    embedding_yes = np.load('./yes.npy')
    label_no = np.load('../label_no.npy')
    label_yes = np.load('../label_yes.npy')

    embedding_no = torch.from_numpy(embedding_no)
    embedding_no = torch.tensor(embedding_no, dtype=torch.float32)

    embedding_yes = torch.from_numpy(embedding_yes)
    embedding_yes = torch.tensor(embedding_yes, dtype=torch.float32)

    label_no = torch.LongTensor(label_no)
    label_no = torch.tensor(label_no)

    label_yes = torch.LongTensor(label_yes)
    label_yes = torch.tensor(label_yes)
    if cuda:
        embedding_no = embedding_no.cuda()
        embedding_yes = embedding_yes.cuda()
        label_no = label_no.cuda()
        label_yes = label_yes.cuda()

    print('Finishing Loading Data')
    return embedding_no, embedding_yes, label_no, label_yes
