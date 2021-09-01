#!/usr/bin/python3

import logging
import numpy as np
from tqdm import tqdm
import os
import sys

from myimage import MyImage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt


FACES_DIR = "newfaces"
DATA_AMOUNT = 10000
IMG_WIDTH = 100
IMG_HEIGHT = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.WARNING) # DEBUG for debugging
print(f"Device: {DEVICE}")


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.c2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.c3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)

        self.c4 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        self.c5 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        self.c6 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        self.c7 = nn.Conv2d(128, 128, kernel_size=2, padding=2)
        self.c8 = nn.Conv2d(128, 128, kernel_size=2, padding=2)
        self.c9 = nn.Conv2d(128, 256, kernel_size=2, stride=2)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.2)

        self.to_linear = None
        x = torch.randn(100, 100).view(-1, 1, 100, 100)
        self.convs(x)

        self.last1 = nn.Linear(self.to_linear, 100)
        self.last2 = nn.Linear(self.to_linear, 100)
        self.last3 = nn.Linear(self.to_linear, 100)
        self.last4 = nn.Linear(self.to_linear, 100)

    def convs(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        x = self.drop(x)
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))
        x = F.relu(self.c6(x))

        x = self.drop(x)
        x = F.relu(self.c7(x))
        x = F.relu(self.c8(x))
        x = F.relu(self.c9(x))

        if self.to_linear is None:
            self.to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        logging.debug(f"self.to_linear: {self.to_linear}")
        x = self.convs(x)
        x = x.view(-1, self.to_linear)

        x1 = self.last1(x)
        logging.debug(f"14: Size: {x1.shape}")
        x2 = self.last2(x)
        x3 = self.last3(x)
        x4 = self.last4(x)

        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x3 = F.softmax(x3, dim=1)
        x4 = F.softmax(x4, dim=1)
        logging.debug(f"15: Size: {x1.shape}")

        return [x1, x2, x3, x4]

class Data:
    def __init__(self, path="training_data.npy", BATCH_SIZE=32, REMAKE_DATA=False):
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

            training_data.append((image.pixels, result))

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
    def __init__(self, net, data, EPOCHS=1, BATCH_SIZE=32):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

        self.net = net
        print(self.net)

        self.data = data

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.05)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.loss_functions = [nn.MSELoss() for _ in range(4)]

        self.train()
        try:
            torch.save(self.net.state_dict(), "models/model.pth")
        except FileNotFoundError:
            os.mkdir("models", mode = 0o666)
            print("Created folder models")



    def train(self):
        losses = []
        idx = 0
        for epoch in tqdm(range(self.EPOCHS)):
            for i in range(0, len(self.data.train_Y), self.BATCH_SIZE):
                batch_X = torch.Tensor(self.data.train_X[i : i + self.BATCH_SIZE]).to(DEVICE).view(
                    -1, 1, 100, 100
                ) / 255.0

                batch_Y = self.data.train_Y[i : i + self.BATCH_SIZE]
                # Convert form from -1, 4, 100 --> 4, -1, 100. Can't use view because you distroy the order
                batch_Y = torch.Tensor([[batch_Y[j][k] for j in range(len(batch_Y))] for k in range(len(batch_Y[0]))]).to(DEVICE) 

                self.net.zero_grad()
                outputs = self.net(batch_X) # Shape: (4, -1, 100)

                loss = 0

                for j in range(len(outputs)):
                    loss += self.loss_functions[j](outputs[j], batch_Y[j])

                losses.append(float(loss))
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
    if sys.argv[1] == "load":
        net.load_state_dict(torch.load("models/model.pth"))
    else:
        net = TrainModel(net, data).net

    with torch.no_grad():
        BATCH_SIZE = 100
        batch_X = torch.Tensor(data.test_X[:BATCH_SIZE]).to(DEVICE).view(-1, 1, 100, 100)
        batch_Y = data.test_Y[:BATCH_SIZE]
        # Convert form from -1, 4, 100 --> 4, -1, 100. Can't use view because you distroy the order
        batch_Y = torch.Tensor([[batch_Y[j][k] for j in range(len(batch_Y))] for k in range(len(batch_Y[0]))]).to(DEVICE) 
        network = net(batch_X)
        print(batch_Y)
        print(network)
        for i in range(4):
            print("##########################", i," ####################")
            for j in range(4):
                print(f"{torch.argmax(network[i][j])} {torch.argmax(batch_Y[i][j])} | diff: {torch.argmax(network[i][j]) - torch.argmax(batch_Y[i][j])}")
