import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class LeNet(Model):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(12, (5, 5), activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = Conv2D(32, (5, 5), activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten = Flatten()

        self.fc1 = Dense(240, activation='relu')
        self.dropout1 = Dropout(0.5)

        self.fc2 = Dense(168, activation='relu')
        self.dropout2 = Dropout(0.5)

        self.fc3 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # 1
        x = self.conv1(inputs)
        x = self.pool1(x)
        # 2
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        return self.fc3(x)
    


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# 参数from_logits表示输入是否为模型的输出结果。如果from_logits为True，则表示传入的输出是模型输出的结果，否则表示传入的是经过softmax激活后的结果。
optmizer = tf.keras.optimizers.Adam()
# 优化器。 Adam 算法，是一种基于梯度下降的优化算法，可以自适应地调整每个参数的学习率。

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = lenet(x, training=True)   # 前向计算
        loss = loss_object(y, logits)   #计算损失

    grads = tape.gradient(loss, lenet.trainable_variables)     # 梯度
    optmizer.apply_gradients(zip(grads, lenet.trainable_variables))     # 梯度和变量打包

    train_loss(loss)
    train_accuracy(y, logits)


@tf.function
def test_step(x, y):
    with tf.GradientTape() as tape:
        logits = lenet(x, training=False)   # 前向计算
        loss = loss_object(y, logits)   #计算损失

    test_loss(loss)
    test_accuracy(y, logits)

def data_generator(x, y, batch_size):
    while True:
        for offset in range(0, len(x), batch_size):
            batch_x = x[offset : offset + batch_size]
            batch_y = y[offset : offset + batch_size]
            yield batch_x, batch_y
        return


data = np.load('data_array64.npz')
train_x, train_y, test_x, test_y = data['train_x'], data['train_y'], data['test_x'], data['test_y']
train_x = train_x.astype('float32') / 255
train_y = train_y.astype('int32')
test_x  = test_x.astype('float32') / 255
test_y = test_y.astype('int32')

# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


batch_size = 32


def training(EPOCHS):
    for epoch in range(EPOCHS):
        # tf.keras.backend.clear_session()
        train_loss.reset_states()   # 重置为0
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        start = time.time()
        for batch_x, batch_y in data_generator(train_x, train_y, batch_size):   
            train_step(batch_x, batch_y)
        end = time.time()
        for batch_x, batch_y in data_generator(test_x, test_y, batch_size): 
            test_step(batch_x, batch_y)

        print(
            f'Epoch: {epoch + 1}\n'  
            f'time {end - start}\n'
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}\n'
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}\n'
            f'------------------------------------------\n'
        )

try:
    lenet = tf.keras.models.load_model('lenet')
except:
    lenet = LeNet(40)

training(600)


save_path = 'lenet'
tf.keras.models.save_model(lenet, save_path)





label_list = []
with open('label_list.txt', 'r') as f:
        while 1:
            str = f.readline().strip()
            if not str:
                  break
            label_list.append(str)

import matplotlib.pyplot as plt
import matplotlib as mpl

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')
prediction_y = lenet.call(test_x[:100])

def plot_image(prediction_arr, true_labels, images):
     plt.figure(figsize=(4, 4))
     # 5 5 -> 25个
     for i in range(len(prediction_arr)):
        img = images[i]
        prediction_label = np.argmax(prediction_arr[i])
        #print(prediction_label, true_labels[i])
        true_label = true_labels[i]
        if prediction_label == true_label:
             color = 'blue'
        else:
             color = 'red'
        plt.subplot(3, 4, i * 2 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.xlabel("{} {:2.0f}% ({})".format(
             label_list[true_labels[i]], \
             100*np.max(prediction_arr[i]), \
             label_list[true_label]
        ), color=color, fontproperties=zhfont)

        plt.subplot(3, 4, i * 2 + 2)
        plt.xticks(range(40))
        plt.yticks([])
        plt.grid(False)
        plt.ylim([0, 1])
        thisplot = plt.bar(range(40), prediction_arr[i], color="#777777")
        thisplot[true_label].set_color('blue')
        thisplot[prediction_label].set_color('red')


plot_image(prediction_y[:5], test_y[:5], test_x[:5])
plt.show()








