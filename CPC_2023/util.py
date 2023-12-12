import collections
import numpy as np


def choose_a_pair(pair, loss_list, vis, args):
    new_pairs = []
    for x in pair:
        if x[0] in loss_list or x[1] in loss_list:
            new_pairs.append(x)
    loss_list = loss_list[-args.num_loss_list:]
    loss_list = loss_list[::-1]

    possible_pair = new_pairs[0]
    if vis[possible_pair[0]] >= args.swap_thresh:
        for x in loss_list:
            if vis[x] < args.swap_thresh:
                for y in new_pairs:
                    if y[0] == x:
                        possible_pair = y
                        break
                break

    vis[possible_pair[0]] += 1
    return possible_pair


def change_one_label(pair, loss_list, old_file, new_file, clean_index, noise_labels, clean_labels, imgs, vis, args):
    onepair = choose_a_pair(pair, loss_list, vis, args)
    s_label, t_label = onepair[0], onepair[1]
    acc = 0
    with open(new_file, 'w') as f:
        for idx, img in enumerate(imgs):
            if idx in clean_index:
                f.write('{} {}\n'.format(img, noise_labels[idx]))
            else:  # noise
                if noise_labels[idx] == s_label:
                    noise_labels[idx] = t_label
                    f.write('{} {}\n'.format(img, noise_labels[idx]))
                else:
                    f.write('{} {}\n'.format(img, noise_labels[idx]))
            if noise_labels[idx] == clean_labels[idx]:
                acc = acc + 1


def sample_selection(loss_matrix, train_file, pairs, vis, args):
    clean_labels, noise_labels, imgs = [], [], []
    with open(train_file, 'r') as f:
        for x in f.readlines():
            x = x.strip().split()
            imgs.append(x[0])
            noise_labels.append(int(x[1]))
    clean_file = './data/' + args.dataset + '/' + args.target + '.txt'
    with open(clean_file, 'r') as f:
        for x in f.readlines():
            x = x.strip().split()
            clean_labels.append(int(x[1]))
    clean_flags = []
    for i in range(len(clean_labels)):
        noisy_label = noise_labels[i]
        clean_label = clean_labels[i]

        if noisy_label == clean_label:
            clean_flags.append(1)
        else:
            clean_flags.append(0)
    loss_sele = loss_matrix[:, :args.train_epochs]
    loss_mean = loss_sele.mean(axis=1)

    loss_list = [0] * args.num_classes
    for i in range(len(noise_labels)):
        loss_list[noise_labels[i]] += loss_mean[i]
    cnt = collections.Counter(noise_labels)
    for i in range(args.num_classes):
        if cnt[i] != 0:
            loss_list[i] /= cnt[i]
    loss_list = np.argsort(loss_list)

    cr = args.clean_rate
    sort_index = np.argsort(loss_mean)

    clean_index = []
    for i in range(int(args.num_classes)):
        c = []
        for idx in sort_index:
            if noise_labels[idx] == i:
                c.append(idx)
        clean_num = int(len(c) * cr)
        clean_idx = c[:clean_num]
        clean_index.extend(clean_idx)
    acc_mum = 0
    for i in clean_index:
        if clean_flags[i] == 1:
            acc_mum += 1
    acc = acc_mum / len(clean_index)
    print("target acc {:.1f}% vs clean rate {}".format(acc * 100, cr))
    selected_file = './data/' + args.dataset + '/' + args.source[0] + args.target[0] + '_selected.txt'
    with open(selected_file, 'w') as f:
        for idx, img in enumerate(imgs):
            if idx in clean_index:
                f.write('{} {}\n'.format(img, noise_labels[idx]))

    change_one_label(pairs, loss_list, train_file, train_file, clean_index, noise_labels, clean_labels, imgs, vis, args)
    return selected_file
