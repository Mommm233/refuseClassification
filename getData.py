import os
from PIL import Image
import numpy as np

dataDir = 'data'
labelDir = 'label'

def fun(dataDir, labelDir):
    x, y = [], []
    lst = os.listdir(dataDir)
    for filename in lst:
        img = Image.open(os.path.join(dataDir, filename))
        img = img.resize((224, 224))
        img = np.array(img)
        x.append(img)
        fil = filename[:-3] + 'txt'
        with open(os.path.join(labelDir, fil), 'r') as f:
            line = f.readline()
            value = line.split(',')
            y.append(value[1].strip())
    return np.array(x), np.array(y)

x, y = fun(dataDir, labelDir)


train_x = x[:int(len(x) * 0.8)]
train_y = y[:int(len(y) * 0.8)]

test_x = x[int(len(x) * 0.8):]
test_y = y[int(len(y) * 0.8):]

np.savez('data_array224.npz', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)