import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=10, help='How many users in one epoch')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    # parser.add_argument('--steps', type=int, default=5, help='Number of step to train.')
    # parser.add_argument('--iterations', type=int, default=100, help='Number of iteration in each step.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training.')
    parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=2, help='Number of hidden unit')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in GRU')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args