import time
import numpy as np
import tensorflow as tf


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(in_channels, kernel_size=3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(in_channels, kernel_size=3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        residual = self.conv1(inputs)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        
        out = self.relu(inputs + residual)
        return out

class ResNet18(tf.keras.Model):
    def __init__(self, resblock,outputs=40):
        super(ResNet18, self).__init__()

        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'),
            resblock(in_channels=64),
            resblock(in_channels=64),
        ])
        
        self.block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, kernel_size=3, strides=(2, 2), padding='same'),
            resblock(in_channels=128),
            resblock(in_channels=128),
        ])
        
        self.block4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, kernel_size=3, strides=(2, 2), padding='same'),
            resblock(in_channels=256),
            resblock(in_channels=256),
        ])
        
        self.block5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, kernel_size=3, strides=(2, 2), padding='same'),
            resblock(in_channels=512),
            resblock(in_channels=512),
        ])
        
        self.block6 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(outputs, activation='softmax')
        
    def call(self, x):
        x = self.block1(x)
        
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.block6(x)
        x = self.fc(x)

        return x



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
        logits = resnet(x, training=True)   # 前向计算
        loss = loss_object(y, logits)   #计算损失

    grads = tape.gradient(loss, resnet.trainable_variables)     # 梯度
    optmizer.apply_gradients(zip(grads, resnet.trainable_variables))     # 梯度和变量打包

    train_loss(loss)
    train_accuracy(y, logits)


@tf.function
def test_step(x, y):
    with tf.GradientTape() as tape:
        logits = resnet(x, training=False)   # 前向计算
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


data = np.load('data_array224.npz')
train_x, train_y, test_x, test_y = data['train_x'], data['train_y'], data['test_x'], data['test_y']
train_x = train_x.astype('float32')
train_y = train_y.astype('int32')
test_x  = test_x.astype('float32')
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
    resnet = tf.keras.models.load_model('resnet')
except:
    resnet = ResNet18(ResidualBlock, 40)



training(2)


save_path = 'resnet'
tf.keras.models.save_model(resnet, save_path)




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
prediction_y = resnet.call(test_x[:32])

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








