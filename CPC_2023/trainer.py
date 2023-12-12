from torch import nn
from torch.autograd import Variable
import torch
import numpy as np


def train(model_instance, train_source_loader, test_source_loader, optimizer, max_epoch, args):
    model_instance.train()
    sample_num = len(test_source_loader.dataset)
    loss_matrix = np.zeros((sample_num, max_epoch))
    epoch = 0
    best_acc = 0
    while True:
        for datas in train_source_loader:
            optimizer.zero_grad()
            inputs_source, labels_source, _ = datas
            inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
            batch_features, output, softmax_output = model_instance(inputs_source)
            total_loss = nn.CrossEntropyLoss()(output, labels_source)
            total_loss.backward()
            optimizer.step()

        eval_result, predict_labels, loss_matrix = evaluate(model_instance, test_source_loader, loss_matrix, epoch)

        print('epoch={}, total_Loss={:.1f}, acc={:.1f}%'.format(epoch, total_loss, eval_result * 100))

        epoch += 1
        if epoch >= max_epoch:
            break
    return loss_matrix


def evaluate(model_instance, input_loader, loss_matrix, epoch):
    model_instance.eval()
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        features, _, probabilities = model_instance(inputs)
        probabilities = probabilities.data.float()
        labels = labels.data.float()
        features = features.data.float()

        if first_test:
            all_probs = probabilities
            all_labels = labels
            all_feats = features
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_feats = torch.cat((all_feats, features), dim=0)
    index = torch.LongTensor(np.array(range(all_labels.size(0)))).cuda()
    a_labels = all_labels
    pred = all_probs[index, a_labels.long()]
    loss = - torch.log(pred)
    loss = loss.data.cpu().numpy()
    loss_matrix[:, epoch] = loss
    _, predict = torch.max(all_probs, dim=1)
    _, top2_values = torch.topk(all_probs, 2, dim=1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).float() / float(all_labels.size()[0])
    predict = predict.data.cpu().numpy().tolist()
    return accuracy, predict, loss_matrix
