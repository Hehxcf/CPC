import argparse
import torch
from torch import optim
from preprocess.data_provider import load_images
import warnings
from Resnet import Resnet50
from trainer import train
from util import sample_selection
from prototype import get_prototype

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='amazon', type=str)
    parser.add_argument('--target', default='dslr', type=str)
    parser.add_argument('--noisy_rate', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--noisy_type', default='uniform', type=str)
    parser.add_argument('--train_epochs', type=int)
    parser.add_argument('--clean_rate', type=float, help='small loss rate')
    parser.add_argument('--loop_prototype', type=int)
    parser.add_argument('--swap_epochs', type=int)
    parser.add_argument('--num_nearest_samples', default=20, type=int)
    parser.add_argument('--num_loss_list', type=int, default=10, help='numbers of the hard classes')
    parser.add_argument('--swap_thresh', type=int, default=2, help='controls the time for swapping for each pair')

    args = parser.parse_args()
    source_file = './data/' + args.dataset + '/' + args.source + '_' + args.noisy_type + '_noisy_' + args.noisy_rate + '.txt'
    target_file = './data/' + args.dataset + '/' + args.target + '.txt'

    if args.dataset == 'Bing-Caltech':
        source_file = './data/Bing-Caltech/' + args.source + '.txt'
        target_file = './data/Bing-Caltech/' + args.target + '.txt'

    if args.dataset == 'office-31':
        args.num_classes = 31
    elif args.dataset == 'office-home':
        args.num_classes = 65
    elif args.dataset == 'Bing-Caltech':
        args.num_classes = 257

    train_epochs = args.train_epochs

    train_source_loader = load_images(source_file, batch_size=32, is_train=False)
    test_source_loader = load_images(target_file, batch_size=32, is_train=False)

    r50 = Resnet50(num_classes=args.num_classes).cuda()
    optimizer = torch.optim.SGD(r50.parameters(), momentum=0.9, lr=args.lr, weight_decay=0.005, nesterov=True)

    new_source_file, prototype, pairs = get_prototype(r50, train_source_loader, test_source_loader, args)

    args.loop_prototype = 1

    train_source_loader = load_images(new_source_file, batch_size=32, is_train=True)
    test_source_loader = load_images(target_file, batch_size=32, is_train=False)

    prototype = torch.from_numpy(prototype).cuda()
    loss_matrix = train(r50, train_source_loader, test_source_loader, optimizer, train_epochs, args)
    vis = [0] * args.num_classes
    _ = sample_selection(loss_matrix, new_source_file, pairs, vis, args)
    train_source_loader = load_images(new_source_file, batch_size=32, is_train=True)

    r50 = Resnet50(num_classes=args.num_classes).cuda()

    for epoch in range(args.swap_epochs):
        optimizer = torch.optim.SGD(r50.parameters(), momentum=0.9, lr=args.lr / (epoch + 1), weight_decay=0.005,
                                    nesterov=True)
        loss_matrix = train(r50, train_source_loader, test_source_loader, optimizer, train_epochs, args)
        if epoch != args.swap_epochs - 1:
            _ = sample_selection(loss_matrix, new_source_file, pairs, vis, args)

        train_source_loader = load_images(new_source_file, batch_size=32, is_train=True)
        test_source_loader = load_images(target_file, batch_size=32, is_train=False)
        if epoch != args.swap_epochs - 1:
            new_source_file, prototype, pairs = get_prototype(r50, train_source_loader, test_source_loader, args)
            prototype = torch.from_numpy(prototype).cuda()
