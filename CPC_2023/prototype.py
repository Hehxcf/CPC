import collections
import torch
from torch.autograd import Variable
import numpy as np


def get_st_feats(model_instance, input_loader):
    model_instance.eval()
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    input_labels = []

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        input_labels.extend(labels.numpy())
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        features, _, _ = model_instance(inputs)
        features = features.data.float()
        if first_test:
            first_test = False
            all_feats = features
        else:
            all_feats = torch.cat((all_feats, features), dim=0)

    all_feats = all_feats.view(-1, 2048).cpu().numpy()
    return all_feats, input_labels


def get_dist_list(prototype, feats, pred, args):
    num_classes = args.num_classes
    dist_list = [[] for _ in range(num_classes)]
    for idx, cls in enumerate(pred):
        minus = abs(np.array(feats[idx]) - prototype[cls])
        distx = np.sum(minus ** 2)
        dist_list[cls].append([distx, idx])
    for i, x in enumerate(dist_list):
        dist_list[i].sort(key=lambda a: a[0], reverse=True)

    return dist_list


def get_prototype(model, train_source_loader, test_source_loader, args):
    all_source_feats, source_labels = get_st_feats(model, train_source_loader)
    all_test_feats, _ = get_st_feats(model, test_source_loader)
    num_classes = args.num_classes
    print('task:', args.source, '->', args.target)

    all_test_feats /= np.linalg.norm(x=all_test_feats, ord=2, axis=1, keepdims=True)
    cnt = collections.Counter(source_labels)

    prototype = np.zeros((num_classes, 2048))
    for i, (sfeat, slabel) in enumerate(zip(all_source_feats, source_labels)):
        prototype[slabel] += all_source_feats[i]

    for i, _ in enumerate(prototype):
        prototype[i] /= cnt[i]

    prototype /= np.linalg.norm(x=prototype, ord=2, axis=1, keepdims=True)

    for i in range(args.loop_prototype):
        sim_matrix = np.matmul(all_test_feats, prototype.transpose())
        pred = np.argmax(sim_matrix, axis=1)

        for idx, cls in enumerate(pred):
            t1 = cls
            prototype[t1] = (prototype[t1] * cnt[t1] + all_test_feats[idx]) / (cnt[t1] + 1)
            cnt[t1] = cnt[t1] + 1

        dist_list = get_dist_list(prototype, all_test_feats, pred, args)
        for i, x in enumerate(dist_list):
            for j in range(len(dist_list[i]) // 5):
                idx = dist_list[i][j][1]
                t1 = i
                prototype[t1] = (prototype[t1] * cnt[t1] - all_test_feats[idx]) / (cnt[t1] - 1)
                cnt[t1] = cnt[t1] - 1
        write_fname = './data/' + args.dataset + '/' + args.source[0] + args.target[0] + '_p.txt'
        write_labels(pred_labels=pred, write_fname=write_fname, args=args)
        pairs, top_two_values = get_switch_pairs(model, train_source_loader, test_source_loader, args)

    return write_fname, prototype, pairs


def get_switch_pairs(model, train_source_loader, test_source_loader, args):
    true_label_list = []
    fname = './data/' + args.dataset + '/' + args.target + '.txt'
    with open(fname, 'r') as f:
        for x in f.readlines():
            true_label_list.append(int(x.split()[-1]))
    f.close()

    all_source_feats, source_labels = get_st_feats(model, train_source_loader)
    all_test_feats, _ = get_st_feats(model, test_source_loader)
    num_classes = args.num_classes
    all_test_feats /= np.linalg.norm(x=all_test_feats, ord=2, axis=1, keepdims=True)
    all_source_feats /= np.linalg.norm(x=all_source_feats, ord=2, axis=1, keepdims=True)
    cnt = collections.Counter(source_labels)

    prototype = np.zeros((num_classes, 2048))
    for i, (sfeat, slabel) in enumerate(zip(all_source_feats, source_labels)):
        prototype[slabel] += all_source_feats[i]
    for i, _ in enumerate(prototype):
        prototype[i] /= cnt[i]

    prototype /= np.linalg.norm(x=prototype, ord=2, axis=1, keepdims=True)
    sim_matrix = np.matmul(all_test_feats, prototype.transpose())
    pred = np.argmax(sim_matrix, axis=1)

    all_feats = all_source_feats
    source_labels.extend(pred.tolist())
    all_labels = source_labels
    sim_matrix = np.matmul(all_test_feats, all_feats.transpose())
    attributes = np.argsort(-sim_matrix, axis=1)[:, :args.num_nearest_samples]
    n, m = sim_matrix.shape
    top_two_values, top_two_values10 = [], []
    for i in range(n):
        top20pred = np.array(all_labels)[attributes[i]]
        counts = np.bincount(top20pred)
        top_values = np.argsort(-counts)[:2]
        if len(top_values) == 1:
            top_values = np.resize(top_values, 2)

        top_two_values.append(top_values.tolist())

    cge_mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(len(pred)):
        cge_mat[top_two_values[i][0]][top_two_values[i][1]] = cge_mat[top_two_values[i][0]][top_two_values[i][1]] + 1

    pair = []
    for a in range(len(cge_mat)):
        for b in range(len(cge_mat)):
            if cge_mat[a][b] + cge_mat[b][a] > 0 and a != b:
                if (b, a) not in pair:
                    pair.append((a, b))

    pair.sort(key=lambda x: min(cge_mat[x[0]][x[1]], cge_mat[x[1]][x[0]]), reverse=True)
    for i, x in enumerate(pair):
        if cge_mat[x[0]][x[1]] / cnt[x[0]] < cge_mat[x[1]][x[0]] / cnt[x[1]]:
            pair[i] = [x[1], x[0]]

    return pair, top_two_values


def write_labels(pred_labels, write_fname, args):
    fname_list = []
    true_label_list = []
    fname = './data/' + args.dataset + '/' + args.target + '.txt'
    with open(fname, 'r') as f:
        for x in f.readlines():
            fname_list.append(x.split()[0])
            true_label_list.append(int(x.split()[-1]))
    f.close()
    cnt = 0
    with open(write_fname, 'w') as ff:
        for i, _ in enumerate(pred_labels):
            # if i not in idx_list:
            ff.write('{} {}\n'.format(fname_list[i], pred_labels[i]))

            if pred_labels[i] == true_label_list[i]:
                cnt = cnt + 1
    with open(write_fname, 'r') as ff:
        len_file = len(ff.readlines())

    print(
        'choose {:.2f}% samples, target acc {:.2f}%'.format(len_file / len(pred_labels) * 100, (cnt / len_file) * 100))
