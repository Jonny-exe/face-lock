#!/usr/bin/python3

import logging
import numpy as np
from tqdm import tqdm
import os
import sys

from myimage import MyImage
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt


FACES_DIR = "newfaces_border"
DATA_AMOUNT = 20000
IMG_WIDTH = 100
IMG_HEIGHT = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.WARNING) # DEBUG for debugging
print(f"Device: {DEVICE}")


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.a1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.a2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.a3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.a4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.a5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.to_linear = None
        x = torch.rand(1, 1, 100, 100)
        self.convs(x)

        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()

        self.l1 = nn.Linear(self.to_linear, 4096)
        self.l2 = nn.Linear(4096, 4096)

        self.last1 = nn.Linear(4096, 100)
        self.last2 = nn.Linear(4096, 100)
        self.last3 = nn.Linear(4096, 100)
        self.last4 = nn.Linear(4096, 100)


    def convs(self, x):
        x = self.pool1(F.relu(self.a1(x), inplace=True))
        x = self.pool2(F.relu(self.a2(x), inplace=True))
        x = F.relu(self.a3(x), inplace=True)
        x = F.relu(self.a4(x), inplace=True)
        x = self.pool3(F.relu(self.a5(x), inplace=True))
        x = self.avgpool(x)
        logging.debug(f"Last conv: {x.shape}")

        if self.to_linear is None:
            self.to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        logging.debug(f"self.to_linear: {self.to_linear}")
        x = self.convs(x)
        x = x.view(-1, self.to_linear)
        x = self.drop1(x)
        x = F.relu(self.l1(x), inplace=True)
        x = self.drop2(x)
        x = F.relu(self.l2(x), inplace=True)

        x1 = self.last1(x)
        # logging.debug(f"14: Size: {x1.shape}")
        # x2 = self.last2(x)
        # x3 = self.last3(x)
        # x4 = self.last4(x)

        x1 = F.softmax(x1, dim=1)
        # x2 = F.softmax(x2, dim=1)
        # x3 = F.softmax(x3, dim=1)
        # x4 = F.softmax(x4, dim=1)
        logging.debug(f"15: Size: {x1.shape}")

        # return [x1, x2, x3, x4]
        return [x1]


class Data:
    def __init__(self, path="training_data.npy", BATCH_SIZE=100, REMAKE_DATA=False):
        if REMAKE_DATA:
            self.make_training_data()
        try:
            self.training_data = np.load(path, allow_pickle=True)
            np.random.shuffle(self.training_data)
        except FileNotFoundError:
            self.training_data = self.make_training_data()

        self.BATCH_SIZE = BATCH_SIZE
        self.format()

    def make_training_data(self):
        training_data = []

        idx = 0
        for filename in tqdm(os.listdir(FACES_DIR)):
            f = os.path.join(FACES_DIR, filename)
            image = MyImage(f)
            if idx == 0:
                image.show_image()
            if len(image.pixels) != 100:
                break

            pieces = [list(map(int, i.split("x"))) for i in filename.split("X")]
            eye = np.eye(100)
            result = [eye[pieces[0][0]], eye[pieces[0][1]], eye[pieces[1][0]], eye[pieces[1][1]]]

            # eye = np.eye(2)

            training_data.append([image.pixels, result])
            # rand = random.randrange(2)
            # img = np.zeros((100, 100)) if rand == 0 else np.full((100,100), 255)
            # training_data.append([img, eye[rand]])

            idx += 1

            if idx == DATA_AMOUNT:
                print("Max amount arrived")
                break
        np.random.shuffle(training_data)
        np.save("training_data", training_data)
        return training_data

    def format(self):
        X = [i[0] for i in self.training_data]
        # .view(-1, 1, 100, 100)
        # FIXME: this may destroy the images. Maybe you have to do manual reshaping
        # X /= 255.0
        # person type data  type people data
        #   100    4  100 ->  4   100   100
        Y = [i[1] for i in self.training_data]

        self.train_X = X[self.BATCH_SIZE :]
        self.test_X = X[: self.BATCH_SIZE]

        self.train_Y = Y[self.BATCH_SIZE :]
        self.test_Y = Y[: self.BATCH_SIZE]


class TrainModel:
    def __init__(self, net, data, EPOCHS=100, BATCH_SIZE=64):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

        self.net = net
        print(self.net)

        self.data = data

        # self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000000, gamma=0.1)
        self.loss_functions = [nn.MSELoss() for _ in range(4)]
        # self.loss_functions = [nn.MultiLabelMarginLoss() for _ in range(4)]
        # self.loss_functions = [nn.L1Loss() for _ in range(4)] 
        self.train()
        try:
            torch.save(self.net.state_dict(), f"models/model{EPOCHS}.pth")
        except FileNotFoundError:
            os.mkdir("models", mode = 0o666)
            print("Created folder models")



    def train(self):
        losses = []
        idx = 0
        for epoch in tqdm(range(self.EPOCHS)):
            for i in tqdm(range(0, len(self.data.train_Y), self.BATCH_SIZE)):
                batch_X = torch.Tensor(self.data.train_X[i : i + self.BATCH_SIZE]).to(DEVICE).view(
                    -1, 1, 100, 100
                ) / 255.0

                # batch_Y = torch.tensor(self.data.train_Y[i : i + self.BATCH_SIZE]).to(DEVICE)
                batch_Y = self.data.train_Y[i : i + self.BATCH_SIZE]
                # Convert form from -1, 4, 100 --> 4, -1, 100. Can't use view because you distroy the order
                batch_Y = torch.Tensor([[batch_Y[j][k] for j in range(len(batch_Y))] for k in range(len(batch_Y[0]))]).to(DEVICE)

                # print(batch_Y[0])
                # print(batch_X[0][0] * 255)
                # plt.imshow(batch_X[0][0].to("cpu") * 255, vmin=0, vmax=255, interpolation='none', cmap="gray")
                # plt.show()

                self.net.zero_grad()
                self.optimizer.zero_grad()
                outputs = self.net(batch_X) # Shape: (4, -1, 100)

                # print(batch_X[0], outputs[0], batch_Y)
                loss = self.loss_functions[0](outputs[0], batch_Y[0])
                # print(loss)
                losses.append(float(loss))
                if i % 100 == 0:
                    print(torch.argmax(batch_Y))
                    print(torch.argmax(outputs[0]))
                    print(float(loss))
                    pass


                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                idx += 0
            if epoch % 10 == 0 and epoch != 0:
                plt.plot(losses)
                plt.show()

        plt.plot(losses)
        plt.show()


if __name__ == "__main__":
    net = Net()
    net.to(DEVICE)
    data = Data()
    if len(sys.argv) > 1:
        net.load_state_dict(torch.load(f"models/model{sys.argv[1]}.pth"))
    else:
        net = TrainModel(net, data).net

    with torch.no_grad():
        BATCH_SIZE = 64
        # batch_X = torch.Tensor(data.test_X[:BATCH_SIZE]).to(DEVICE).view(-1, 1, 100, 100)
        batch_X = torch.Tensor(data.train_X[:BATCH_SIZE]).to(DEVICE).view(-1, 1, 100, 100)
        batch_Y = data.train_Y[:BATCH_SIZE]
        # Convert form from -1, 4, 100 --> 4, -1, 100. Can't use view because you distroy the order
        batch_Y = torch.Tensor([[batch_Y[j][k] for j in range(len(batch_Y))] for k in range(len(batch_Y[0]))]).to(DEVICE) 
        network = net(batch_X)
        print(f"network: {network[0][0][39]}, {network[0][1][39]}, {network[0][2][39]}")
        print(network[0].shape)
        for i in range(len(network)):
            print(f"{network[i]} || {batch_Y[i]}")
            print(f"{torch.argmax(network[i])} || {torch.argmax(batch_Y[i])}")
        for i in range(10):
            print("##########################", i," ####################")
            # for j in range(4):
            #     print(f"{torch.argmax(network[i][j])} {torch.argmax(batch_Y[i][j])} | diff: {torch.argmax(network[i][j]) - torch.argmax(batch_Y[i][j])}")
                # print(f"{torch.argmax(network[i][j])} {torch.argmax(batch_Y[i][j])} | diff: {torch.argmax(network[i][j]) - torch.argmax(batch_Y[i][j])}")
            print(f"{torch.argmax(network[0][i])} {torch.argmax(batch_Y[0][i])} | diff: {torch.argmax(network[0][i]) - torch.argmax(batch_Y[0][i])}")

