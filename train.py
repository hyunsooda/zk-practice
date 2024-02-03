from pathlib import Path
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    print("Reading the data...")
    data = pd.read_csv('input/train.csv', sep=",")
    test_data = pd.read_csv('input/test.csv', sep=",")

    print("Reshaping the data...")
    dataFinal = data.drop('label', axis=1)
    labels = data['label']

    dataNp = dataFinal.values
    labelsNp = labels.values
    test_dataNp = test_data.values

    print("Data is ready")

    return dataNp, labels, labelsNp, test_dataNp

def show_data(dataNp, labels, test_dataNp, display=False):
    plt.figure(figsize=(14, 12))

    pixels = dataNp[10].reshape(28, 28)
    plt.subplot(321)
    sns.heatmap(data=pixels)

    pixels = dataNp[11].reshape(28, 28)
    plt.subplot(322)
    sns.heatmap(data=pixels)

    pixels = dataNp[20].reshape(28, 28)
    plt.subplot(323)
    sns.heatmap(data=pixels)

    pixels = dataNp[32].reshape(28, 28)
    plt.subplot(324)
    sns.heatmap(data=pixels)

    pixels = dataNp[40].reshape(28, 28)
    plt.subplot(325)
    sns.heatmap(data=pixels)

    pixels = dataNp[52].reshape(28, 28)
    plt.subplot(326)
    sns.heatmap(data=pixels)

    if display:
        print(labels[10], " / ", labels[11])
        print(labels[20], " / ", labels[32])
        print(labels[40], " / ", labels[52])

        plt.show()

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, q=False):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, output_size)

        self.q = q
        if q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        x = F.log_softmax(x)
        if self.q:
            x = self.dequant(x)
        return x

def train(dataNp, labelsNp, input_size, hidden_size, output_size, learning_rate, batch_size, epoch, display=False):
    x = torch.FloatTensor(dataNp.tolist())
    y = torch.LongTensor(labelsNp.tolist())

    net = Network(input_size, hidden_size, output_size)
    print(net)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    loss_log = []

    for e in range(epoch):
        for i in range(0, x.shape[0], batch_size):
            x_mini = x[i:i + batch_size]
            y_mini = y[i:i + batch_size]

            x_var = Variable(x_mini)
            y_var = Variable(y_mini)

            optimizer.zero_grad()
            net_out = net(x_var)

            loss = loss_func(net_out, y_var)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                loss_log.append(loss.item())
        print('Epoch: {} - Loss: {:.6f}'.format(e, loss.item()))

    if display:
        plt.figure(figsize=(10,8))
        plt.plot(loss_log)
        plt.show()

    return net

def validate(net, test_dataNp, display=False):
    test = torch.FloatTensor(test_dataNp.tolist())
    test_var = Variable(test)

    net_out = net(test_var)

    print(torch.max(net_out.data, 1)[1].numpy())

    plt.figure(figsize=(14, 12))

    pixels = test_dataNp[1].reshape(28, 28)
    plt.subplot(321)
    sns.heatmap(data=pixels)
    test_sample = torch.FloatTensor(test_dataNp[1].tolist())
    test_var_sample = Variable(test_sample)
    net_out_sample = net(test_var_sample)

    pixels = test_dataNp[10].reshape(28, 28)
    plt.subplot(322)
    sns.heatmap(data=pixels)
    test_sample = torch.FloatTensor(test_dataNp[10].tolist())
    test_var_sample = Variable(test_sample)
    net_out_sample = net(test_var_sample)

    pixels = test_dataNp[20].reshape(28, 28)
    plt.subplot(323)
    sns.heatmap(data=pixels)
    test_sample = torch.FloatTensor(test_dataNp[20].tolist())
    test_var_sample = Variable(test_sample)
    net_out_sample = net(test_var_sample)

    pixels = test_dataNp[30].reshape(28, 28)
    plt.subplot(324)
    sns.heatmap(data=pixels)
    test_sample = torch.FloatTensor(test_dataNp[30].tolist())
    test_var_sample = Variable(test_sample)
    net_out_sample = net(test_var_sample)

    pixels = test_dataNp[100].reshape(28, 28)
    plt.subplot(325)
    sns.heatmap(data=pixels)
    test_sample = torch.FloatTensor(test_dataNp[100].tolist())
    test_var_sample = Variable(test_sample)
    net_out_sample = net(test_var_sample)

    pixels = test_dataNp[2000].reshape(28, 28)
    plt.subplot(326)
    sns.heatmap(data=pixels)
    test_sample = torch.FloatTensor(test_dataNp[2000].tolist())
    test_var_sample = Variable(test_sample)
    net_out_sample = net(test_var_sample)

    if display:
        plt.show()
        print("Prediction: {} / {}".format(torch.max(net_out.data, 1)[1].numpy()[1], torch.max(net_out.data, 1)[1].numpy()[10]))
        print("Prediction: {} / {}".format(torch.max(net_out.data, 1)[1].numpy()[20], torch.max(net_out.data, 1)[1].numpy()[30]))
        print("Prediction: {} / {}".format(torch.max(net_out.data, 1)[1].numpy()[100], torch.max(net_out.data, 1)[1].numpy()[2000]))

    output = (torch.max(net_out.data, 1)[1]).numpy()
    np.savetxt("out.csv", np.dstack((np.arange(1, output.size+1),output))[0],"%d,%d",header="ImageId,Label")

def load_model(input_size, hidden_size, output_size, model_path):
    net = Network(input_size, hidden_size, output_size)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net

def fuse_modules(net):
    torch.quantization.fuse_modules(net, [['l1', 'relu']], inplace=True)

def save_weights(net, scale, scaleFactor):
    Path("weight").mkdir(parents=True, exist_ok=True)
    for (file_path, param) in zip(["weight/fc1.txt", "weight/relu1.txt", "weight/fc2.txt", "weight/softmax.txt"], net.parameters()):
        print(param.shape)
        tensor = param.detach()
        if scale == True:
            tensor = torch.ceil(tensor * scaleFactor)
        np.savetxt(file_path, tensor, fmt='%s', delimiter=',')

def model_test(input_size, hidden_size, output_size, model_path, display=False):
    dataNp, labels, labelsNp, test_dataNp = load_data();
    net = load_model(input_size, hidden_size, output_size, model_path)

    test = torch.FloatTensor(test_dataNp.tolist())
    test_var = Variable(test)
    target_idx = random.randint(0, test_var.shape[0]) 
    target_image = test_var[target_idx]
    test_out = torch.argmax(net.forward(target_image))

    if display == True:
        plt.figure(figsize=(14, 12))
        pixels = target_image.reshape(28, 28)
        plt.subplot(321)
        sns.heatmap(data=pixels)
        plt.show()
    
    print("target idx: {}, test out: {}".format(target_idx, test_out))
    return target_idx, test_out

def model_train(input_size, output_size, hidden_size, epoch, batch_size, learning_rate, model_path, scale, scaleFactor):
    dataNp, labels, labelsNp, test_dataNp = load_data();
    net = train(dataNp, labelsNp, input_size, hidden_size, output_size, learning_rate, batch_size, epoch)
    validate(net, test_dataNp)
    torch.save(net.state_dict(), model_path)
    save_weights(net, scale, scaleFactor)

def main():
    # hyperparameters
    input_size = 784
    output_size = 10
    hidden_size = 200
    epoch = 20
    batch_size = 50
    learning_rate = 0.00005
    scaleFactor = 1000
    model_path = "model.pt"

    parser = argparse.ArgumentParser(description="Argument parser example")
    parser.add_argument("option", choices=["train", "test"], help="Specify an option (train or test)")

    args = parser.parse_args()
    if args.option == "train":
        model_train(input_size, output_size, hidden_size, epoch, batch_size, learning_rate, model_path, True, scaleFactor)
    elif args.option == "test":
        target_idx, test_out = model_test(input_size, hidden_size, output_size, model_path)
        with open('test-result.txt', 'w') as fp:
            fp.write("{}:{}".format(target_idx, test_out))

main()
