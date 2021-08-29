#!/usr/bin/python3
from PIL import Image
import matplotlib.pyplot as mpl

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

    def show_image(self):
        mpl.imshow(self.pixels, cmap="gray")
        mpl.show()

if __name__ == "__main__":
    image = MyImage("random-image")
    image.show_image()
