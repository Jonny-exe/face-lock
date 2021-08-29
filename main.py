#!/usr/bin/python3

import numpy as np
from tqdm import tqdm
import os
from myimage import MyImage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as mpl


FACES_DIR = "newfaces"
DATA_AMOUNT = 9000
IMG_WIDTH = 100
IMG_HEIGHT = 100
REMAKE_DATA = False


def make_training_data():
    training_data = []
    idx = 0
    for filename in tqdm(os.listdir(FACES_DIR)):
        f = os.path.join(FACES_DIR, filename)
        image = MyImage(f)
        if len(image.pixels) != 100:
            break

        pieces = [list(map(int, i.split("x"))) for i in filename.split("X")]
        result = [pieces[0][0], pieces[0][1], pieces[1][0], pieces[1][1]]
        training_data.append((image.pixels, result))
        idx += 1

        if idx == DATA_AMOUNT:
            print("Max amount arrived")
            break
    np.random.shuffle(training_data)
    np.save("training_data", training_data)


if REMAKE_DATA:
    print("Start")
    make_training_data()



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)

        self.to_linear = None
        x = torch.randn(100, 100).view(-1, 1, 100, 100)
        self.convs(x)

        self.fc1 = nn.Linear(self.to_linear, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)
        self.fc4 = nn.Linear(4, 4)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))

        if self.to_linear is None:
            self.to_linear = x[0].shape[1] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


net = Net()

print(net)



optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()
training_data = np.load("training_data.npy", allow_pickle=True)
print("Length: ", len(training_data[0][0]))
X = torch.Tensor([i[0] for i in training_data]).view(-1, 1, 100, 100)
X /= 255.0
Y = torch.Tensor([i[1] for i in training_data])
Y /= 100.0

train_X = X[100:]
test_X = X[:100]

train_Y = Y[100:]
test_Y = Y[:100]

BATCH_SIZE = 100
EPOCHS = 1


for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_Y), BATCH_SIZE)):
        batch_X = train_X[i : i + BATCH_SIZE].view(-1, 1, 100, 100)
        batch_Y = train_Y[i : i + BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        print(
            f"outputs: {outputs.shape}\nbatch_Y: {batch_Y.shape}\nbatch_X: {batch_X.shape}"
        )
        loss = loss_function(outputs, batch_Y)
        loss.backward()
        optimizer.step()
    print("Loss: ", loss)


class MyImage:
    def __init__(self, image_path):
        self.path = image_path
        self.get_image_pixels()

    def get_image_pixels(self):
        i = Image.open(self.path).convert("L")
        p = i.load()
        self.width, self.height = i.size
        self.pixels = [[0 for _ in range(self.width)] for _ in range(self.height)]

        for x in range(self.width):
            for y in range(self.height):
                cpixel = p[x, y]
                self.pixels[y][x] = cpixel


image = MyImage("random-image")
print(len(image.pixels), len(image.pixels[0]))
mpl.imshow(image.pixels, cmap="gray")
mpl.show()
