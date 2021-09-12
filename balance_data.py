#!/usr/bin/python3
import os
FACES_DIR = "newsingle"
def check_data_is_balanced():
    data_sum = [0, 0, 0, 0]
    idx = 0
    for filename in os.listdir(FACES_DIR):
        # f = os.path.join(FACES_DIR, filename)
        data = [x.split("x") for x in filename.split("X")]
        data = [data[0][0], data[0][1], data[1][0], data[1][1]]
        data = list(map(int, data))

        for i in range(4):
            data_sum[i] += data[i]
        idx += 1

    for i in range(4):
        data_sum[i] /= idx
    return data_sum
    
if __name__ == "__main__":
    mean_data = check_data_is_balanced()
    print(mean_data)



