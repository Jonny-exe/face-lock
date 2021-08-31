#!/usr/bin/python3

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
import matplotlib.pyplot as mpl


FACES_DIR = "newfaces"
DATA_AMOUNT = 10000
IMG_WIDTH = 100
IMG_HEIGHT = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.a1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=2, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=2, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=2)
        self.d2 = nn.Conv2d(128, 128, kernel_size=2)
        self.d3 = nn.Conv2d(128, 10, kernel_size=2)

        self.to_linear = None
        x = torch.randn(100, 100).view(-1, 1, 100, 100)
        self.convs(x)
        self.last = nn.Linear(self.to_linear, 100)

    def convs(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        # 4x4
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        # 2x2
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        # 1x128
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        if self.to_linear is None:
            self.to_linear = x[0].shape[1] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.to_linear)

        x1 = self.last(x)
        x2 = self.last(x)
        x3 = self.last(x)
        x4 = self.last(x)

        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x3 = F.softmax(x3, dim=1)
        x4 = F.softmax(x4, dim=1)

        return [x1, x2, x3, x4]

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

            training_data.append((image.pixels, result))

            idx += 1

            if idx == DATA_AMOUNT:
                print("Max amount arrived")
                break
        np.random.shuffle(training_data)
        np.save("training_data", training_data)
        return training_data

    def format(self):
        X = torch.Tensor([i[0] for i in self.training_data]).view(-1, 1, 100, 100).to(DEVICE)
        X /= 255.0
        # person type data  type people data
        #   100    4  100 ->  4   100   100
        Y = torch.Tensor([i[1] for i in self.training_data]).to(DEVICE)
        # Y = [[y[i][k] for k in range(len(y[i]))] for i in range(len(y[0]))]

        self.train_X = X[self.BATCH_SIZE :]
        self.test_X = X[: self.BATCH_SIZE]

        self.train_Y = Y[self.BATCH_SIZE :]
        self.test_Y = Y[: self.BATCH_SIZE]


class TrainModel:
    def __init__(self, net, data, EPOCHS=13, BATCH_SIZE=100):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

        self.net = net
        print(self.net)

        self.data = data

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.loss_functions = [nn.MSELoss() for _ in range(4)]

        self.train()
        try:
            torch.save(self.net.state_dict(), "models/model.pth")
        except FileNotFoundError:
            os.mkdir("models", mode = 0o666)
            print("Created folder models")



    def train(self):
        for epoch in tqdm(range(self.EPOCHS)):
            for i in range(0, len(self.data.train_Y), self.BATCH_SIZE):
                batch_X = self.data.train_X[i : i + self.BATCH_SIZE].view(
                    -1, 1, 100, 100
                )

                batch_Y = self.data.train_Y[i : i + self.BATCH_SIZE]
                # Convert form from -1, 4, 100 --> 4, -1, 100. Can't use view because you distroy the order
                batch_Y = torch.Tensor([[batch_Y[j][k].tolist() for j in range(len(batch_Y))] for k in range(len(batch_Y[0]))]).to(DEVICE) 

                self.net.zero_grad()
                outputs = self.net(batch_X) # Shape: (4, -1, 100)

                loss = 0

                for j in range(len(outputs)):
                    loss += self.loss_functions[j](outputs[j], batch_Y[j])

                loss.backward()
                self.optimizer.step()


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
        batch_X = data.test_X[:BATCH_SIZE].view(-1, 1, 100, 100)
        batch_Y = data.test_Y[:BATCH_SIZE]
        # Convert form from -1, 4, 100 --> 4, -1, 100. Can't use view because you distroy the order
        batch_Y = torch.Tensor([[batch_Y[j][k].tolist() for j in range(len(batch_Y))] for k in range(len(batch_Y[0]))]).to(DEVICE) 
        network = net(batch_X)
        print(batch_Y)
        print(network)
        for i in range(4):
            print("##########################", i," ####################")
            for j in range(2):
                print(f"Network  Argmax: {torch.argmax(network[i][j])}")
                print(f"Expected Argmax: {torch.argmax(batch_Y[i][j])}")
