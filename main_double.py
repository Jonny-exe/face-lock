#!/usr/bin/python3

from video import play_webcam
from eprint import eprint
import subprocess
import argparse
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
import cv2 as cv


FACES_DIR = "newsingle"
DATA_AMOUNT = 5000
IMG_WIDTH = 100
IMG_HEIGHT = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.WARNING)  # DEBUG for debugging
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

    def convs(self, x):
        x = self.pool1(F.relu(self.a1(x), inplace=True))
        x = F.relu(self.a2(x))
        x = self.pool2(F.relu(self.a3(x), inplace=True))
        x = F.relu(self.a4(x))
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
        x = F.relu(self.l1(self.drop1(x)), inplace=True)
        x = F.relu(self.l2(self.drop2(x)), inplace=True)

        x1 = self.last1(x)
        x2 = self.last2(x)

        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        logging.debug(f"15: Size: {x1.shape}")

        return [x1, x2]


class Data:
    def __init__(self, path="training_data/training_data.npy", BATCH_SIZE=10, REMAKE_DATA=False):
        if REMAKE_DATA:
            self.make_training_data()
        try:
            self.training_data = np.load(
                f"training_data/training_data_{FACES_DIR}.npy", allow_pickle=True
            )
            np.random.shuffle(self.training_data)
        except FileNotFoundError:
            # os.mkdir("training_data")
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
                image.show()
            if len(image.pixels) != 100:
                break

            pieces = [list(map(int, i.split("x"))) for i in filename.split("X")]
            eye = np.eye(100)
            result = [
                eye[pieces[0][0]],
                eye[pieces[0][1]],
                eye[pieces[1][0]],
                eye[pieces[1][1]],
            ]

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
        np.save(f"training_data/training_data_{FACES_DIR}", training_data)
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
    def __init__(
        self,
        net,
        data,
        STARTING_EPOCHS=0,
        EPOCHS=32,
        turn=2,
        # BATCH_SIZE=300,
        BATCH_SIZE=4,
        optimizer_state=None,
        loss=None,
        save=None,
        GRAPH=False,
    ):
        assert turn == 2 or turn == 0
        self.turn = turn
        self.EPOCHS = EPOCHS
        self.STARTING_EPOCHS = STARTING_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.GRAPH = GRAPH

        self.net = net

        self.data = data

        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000000, gamma=0.1
            # self.optimizer, step_size=30, gamma=0.1
        )

        self.loss_functions = [nn.MSELoss().to(DEVICE) for _ in range(4)]

        loss = self.train()
        data_to_save = {
            "net": self.net.state_dict(),
            "epochs": self.EPOCHS,
            "optimizer": self.optimizer.state_dict(),
            "loss": loss,
        }
        try:
            torch.save(
                data_to_save, f"models/model{EPOCHS}.pth" if save is None else save
            )
        except FileNotFoundError:
            os.mkdir("models", mode=0o666)
            print("Created folder models")
            torch.save(
                data_to_save, f"models/model{EPOCHS}.pth" if save is None else save
            )

    def train(self):
        losses = []
        idx = 0
        loss = 0
        for epoch in tqdm(range(self.STARTING_EPOCHS, self.EPOCHS, 1)):
            for i in tqdm(range(0, len(self.data.train_Y), self.BATCH_SIZE)):



                # batch_Y = torch.tensor(self.data.train_Y[i : i + self.BATCH_SIZE]).to(DEVICE)
                batch_Y = self.data.train_Y[i : i + self.BATCH_SIZE]
                # Convert form from -1, 4, 100 --> 4, -1, 100. Can't use view because you distroy the order
                batch_Y = torch.Tensor(
                    [
                        [batch_Y[j][k] for j in range(len(batch_Y))]
                        for k in range(len(batch_Y[0]))
                    ]
                ).to(DEVICE)



                batch_X = (
                    torch.Tensor(self.data.train_X[i : i + self.BATCH_SIZE])
                    .to(DEVICE)
                    .view(-1, 1, 100, 100)
                    # / 255.0
                )

                if idx == 0:
                    print(torch.argmax(batch_Y[0][0]))
                    print(torch.argmax(batch_Y[1][0]))
                    print(torch.argmax(batch_Y[2][0]))
                    print(torch.argmax(batch_Y[3][0]))
                    print(np.array(batch_X[0][0].to("cpu")))
                    plt.imshow(np.array(batch_X[0][0].to("cpu")), cmap="gray")
                    plt.show()
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
                batch_X /= 255.0


                self.net.zero_grad()
                self.optimizer.zero_grad()
                outputs = self.net(batch_X)  # Shape: (4, -1, 100)

                loss = 0
                for j in range(len(outputs)):
                    loss += self.loss_functions[j](outputs[j], batch_Y[j+self.turn])
                    if idx == 0 or idx == 1000 or idx == 4000 or idx == 6000 or idx == 5000 or idx == 10000 or idx == 8000:
                        print(torch.argmax(outputs[j][0]), outputs[j][0][torch.argmax(outputs[j][0])])
                        print(torch.argmax(batch_Y[j][0]), batch_Y[j][0][torch.argmax(batch_Y[j][0])])
                        print(loss)


                assert loss > 0 and loss < 1

                losses.append(float(loss))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                idx += 1
            if epoch % 10 == 0 and epoch != 0 and self.GRAPH:
                plt.plot(losses)
                plt.show()

        plt.plot(losses)
        plt.show()
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--retrain", type=bool)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--testsingle", type=str)
    parser.add_argument("--testall", type=bool, default=False)
    parser.add_argument("--remakedata", type=bool, default=False)
    parser.add_argument("--testlive", type=bool, default=False)

    args = parser.parse_args()

    net = Net()
    net.to(DEVICE)
    data = Data(REMAKE_DATA=args.remakedata)
    if args.load is not None:
        model_data = torch.load(args.load)
        print(model_data.keys())
        net.load_state_dict(model_data["net"])
        print(f"This will be EPOCH = {model_data['epochs'] + args.epochs}")
        if args.retrain is not None:
            net = TrainModel(
                net,
                data,
                EPOCHS=args.epochs + model_data["epochs"],
                STARTING_EPOCHS=model_data["epochs"],
                optimizer_state=model_data["optimizer"],
                save=args.save,
            ).net
    else:
        net_x = TrainModel(net, data, EPOCHS=args.epochs, save=args.save, turn=0).net
        net_y = TrainModel(net, data, EPOCHS=args.epochs, save=args.save, turn=2).net
    print(f"This will be saved in f{args.save}")

    with torch.no_grad():
        BATCH_SIZE = 64
        # batch_X = torch.Tensor(data.test_X[:BATCH_SIZE]).to(DEVICE).view(-1, 1, 100, 100)
        batch_X = (
            torch.Tensor(data.test_X[:BATCH_SIZE]).to(DEVICE).view(-1, 1, 100, 100)
        )
        batch_Y = data.test_Y[:BATCH_SIZE]
        # Convert form from -1, 4, 100 --> 4, -1, 100. Can't use view because you distroy the order
        batch_Y = torch.Tensor(
            [
                [batch_Y[j][k] for j in range(len(batch_Y))]
                for k in range(len(batch_Y[0]))
            ]
        ).to(DEVICE)

        out_x = net_x(batch_X)
        out_y = net_y(batch_X)
        outputs = [out_x[0], out_x[1], out_y[0], out_y[1]]
        total_loss = 0

        if args.testlive:
            play_webcam(net, DEVICE)

        if args.verbose:
            idx = 0
            for i in range(10):
                for x in range(4):
                    # print(
                    #     f"{torch.argmax(network[x][i])} {torch.argmax(batch_Y[x][i])} | diff: {torch.argmax(network[x][i]) - torch.argmax(batch_Y[x][i])}"
                    # )
                    total_loss += abs(torch.argmax(outputs[x][i]) - torch.argmax(batch_Y[x][i]))
                    idx += 1
            print(f"Mean error:  {total_loss / idx}")


            # for filename in os.listdir(FACES_DIR):

            if args.testsingle is not None:
                # f = f"{FACES_DIR}/{torch.argmax(batch_Y[0][i])}x{torch.argmax(batch_Y[1][i])}X{torch.argmax(batch_Y[2][i])}x{torch.argmax(batch_Y[3][i])}"
                f = args.testsingle
                # f = os.path.join(FACES_DIR, filename)
                image = torch.Tensor(MyImage(f).pixels).view(1, 1, 100, 100).to(DEVICE)
                output = net(image)
                arguments = [str(int(torch.argmax(x))) for x in output]

                subprocess.run(
                        [
                            "./draw_border.sh",
                            f,
                            arguments[0],
                            arguments[1],
                            arguments[2],
                            arguments[3],
                            ]
                        )
                output_image = MyImage("output_image.jpeg").show()

            if args.testall:
                for i in range(10):
                    f = f"{FACES_DIR}/{torch.argmax(batch_Y[0][i])}x{torch.argmax(batch_Y[1][i])}X{torch.argmax(batch_Y[2][i])}x{torch.argmax(batch_Y[3][i])}"
                    # f = os.path.join(FACES_DIR, filename)
                    image = torch.Tensor(MyImage(f).pixels).view(1, 1, 100, 100).to(DEVICE)
                    out_x = net_x(image)
                    out_y = net_y(image)
                    output = [out_x[0], out_x[1], out_y[0], out_y[1]]
                    arguments = [str(int(torch.argmax(x))) for x in output]

                    subprocess.run(
                        [
                            "./draw_border.sh",
                            f,
                            arguments[0],
                            arguments[1],
                            arguments[2],
                            arguments[3],
                        ]
                    )
                    output_image = MyImage("output_image.jpeg").show()
                    subprocess.run(["rm", "output_image.jpeg"])
