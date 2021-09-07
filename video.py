#!/usr/bin/python3
import cv2 as cv
import torch
from copy import deepcopy
import numpy as np

def play_webcam(net, DEVICE):
    capture = cv.VideoCapture(0)
    try:
        while True:
            isTrue, frame = capture.read()
            frame = cv.resize(frame, (100, 100)) 
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Grayscale

            frame_copy = deepcopy(frame)
            frame_copy = torch.Tensor(frame_copy).to(DEVICE).view(1, 1, 100, 100)

            output = net(frame_copy)
            output = [int(torch.argmax(x)) for x in output]

            x, y, width, height = output[0], output[1], output[2], output[3]

            frame = cv.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv.imshow("Video", frame)
            """

            cv.imshow("Video", frame)
            """
            cv.waitKey(200)
    except KeyboardInterrupt:
        pass

    capture.release()
    cv.destroyAllWindows()

