#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: main.py
@time: 2018/10/16 18:58
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import argparse

from dataHelper import load_mnist_data, MnistDataSet
from models import MultiLayersNetwork
from myutils import write_data_to_file, write_to_csv, calc_acc

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


def train(train_dataloaders, model, criterion, optimizer, epoch):
    """
    train the model for one epoch!
    :param train_dataloaders:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :return:
    """
    train_loss = []
    train_acc = []
    model.train()
    for i, sample in enumerate(train_dataloaders):
        input = Variable(sample['data']).to(device)
        target = Variable(sample['label']).view(-1).to(device)

        # print("target size: ", target.size())
        # compute output
        output = model(input)
        loss = criterion(output, target)
        acc = accuracy(output, target)

        # compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_acc.append(acc)

        if i % 10000 == 0:
            print("epoch: {}, iteration: {}, loss: {}, acc: {}".format(epoch, i,  loss, acc))
    return np.mean(train_loss), np.mean(train_acc)


def evaluate(valid_dataloaders, model, criterion, epoch):
    valid_loss = []
    valid_acc = []
    model.eval()
    for i, sample in enumerate(valid_dataloaders):
        input = Variable(sample['data']).to(device)
        target = Variable(sample['label']).view(-1).to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        acc = accuracy(output, target)

        valid_loss.append(loss.item())
        valid_acc.append(acc)
    return np.mean(valid_loss), np.mean(valid_acc)


def accuracy(output, target):
    """Compute the accuracy.
    :param output: [b, c]
    :param target: [b]
    :return:
    """
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, 1)
        correct = pred.eq(target)
        acc = correct.view(-1).float().sum() / batch_size
    return acc


def main(args):
    # load data
    print("load data......")
    train_datas = MnistDataSet(data_type="train")
    valid_datas = MnistDataSet(data_type="valid")
    train_dataloaders = DataLoader(train_datas, batch_size=args.batch_size, shuffle=True)
    valid_dataloaders = DataLoader(valid_datas, batch_size=args.batch_size, shuffle=True)
    print("train data size: ", len(train_datas))
    print("valid data size: ", len(valid_datas))

    # create models
    print("create models....")
    model = nn.Sequential(
        nn.Linear(81, 500),
        nn.BatchNorm1d(500),
        nn.Dropout(p=args.drop_out),
        nn.ReLU(),
        nn.Linear(500, 300),
        nn.BatchNorm1d(300),
        nn.Dropout(p=args.drop_out),
        nn.ReLU(),
        nn.Linear(300, 10)
    )
    # model = MultiLayersNetwork(81, 100, 60, 10)
    print(model)
    model.to(device)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.optim == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optim == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), args.lr,
                                         weight_decay=args.weight_decay)

    # begin train
    # data to plot
    loss_dict = {"train_loss": [], "valid_loss": []}
    acc_dict = {"train_acc": [], "valid_acc": []}
    best_train_acc, best_valid_acc = 0., 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train_loss, train_acc = train(train_dataloaders, model, criterion, optimizer, epoch)
        print("Epoch: {} done, loss: {}, train acc: {}".format(epoch, train_loss, train_acc))
        # valid for one epoch
        valid_loss, valid_acc = evaluate(valid_dataloaders, model, criterion, epoch)
        print("valid the model: loss: {}, valid acc: {}".format(valid_loss, valid_acc))

        # save data to plot
        loss_dict["train_loss"].append(train_loss)
        loss_dict["valid_loss"].append(valid_loss)
        acc_dict["train_acc"].append(train_acc)
        acc_dict["valid_acc"].append(valid_acc)

        # save current best model(s.t. valid acc)
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            best_train_acc = train_acc
            torch.save(model.state_dict(), args.model_path)
    # train done
    print("train is done!")
    print("for the best model, train acc is: {}, valid acc is: {}".format(best_train_acc, best_valid_acc))
    # write data to file
    write_data_to_file(loss_dict, "result/loss_dict.pkl")
    write_data_to_file(acc_dict, "result/acc_dict.pkl")


def predict(args):
    # load test data
    print("load test data....")
    test_datas = MnistDataSet(file_name='./data/test_data/TestSamples.csv',
                              data_type="test")
    test_dataloader = DataLoader(test_datas, batch_size=1)
    print("training data size: ", len(test_datas))

    # load model
    print("loading models....")
    model = nn.Sequential(
        nn.Linear(81, 500),
        nn.BatchNorm1d(500),
        nn.Dropout(p=args.drop_out),
        nn.ReLU(),
        nn.Linear(500, 300),
        nn.BatchNorm1d(300),
        nn.Dropout(p=args.drop_out),
        nn.ReLU(),
        nn.Linear(300, 10)
    )
    model.to(device)
    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    # set to eval mode
    model.eval()

    res_list = []
    print("begin predict....")
    for idx, sample in enumerate(test_dataloader):
        data = Variable(sample['data']).to(device)

        output = model(data)
        predict = torch.argmax(output)
        # print("predict: ", predict.item())
        res_list.append(predict.item())

    # write res to csv
    write_to_csv(res_list, args.predict_path)
    print("write done!")

    # calc acc.
    # acc = calc_acc(args.predict_path, 'data/test_data/TestLabels.csv')
    # print("acc is: {}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch fine tune MNIST data Training')

    # mode params
    parser.add_argument("--train", dest='train', action='store_true',
                        help='use model to do train')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--predict', dest='predict', action='store_true',
                        help='use  model to do prediction')

    # model params
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run (default: 50)')
    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning_rate', default=0.00001, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--drop_out', default=0.1, type=float,
                        help='drop out rate')
    parser.add_argument("--optim", '--op', default='momentum', type=str,
                        help='use what optimizer (default: momentum)')

    # log params
    parser.add_argument('--print_freq', '-p', default=104, type=int,
                        metavar='N', help='print frequency (default: 104 batch)')

    # save params
    parser.add_argument("--model_path", default="result/best_model.pkl",
                        help="current best model path")

    parser.add_argument('--test_dir', default='test_b', type=str,
                        help='test data dir (default: test_b)')
    parser.add_argument("--predict_path", default='data/test_data/Result.csv', type=str,
                        help='test data predict file path')

    args = parser.parse_args()
    predict(args)
    # if args.train:
    #     main(args)
    # elif args.predict:
    #     predict(args)
    # else:
    #     print("invalid args")




