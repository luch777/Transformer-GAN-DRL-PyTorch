import os, time, pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import scipy.io as sio
import tensorflow.contrib.slim as slim
from config import args

tf.reset_default_graph()
select_number = args.select_number
# training parameters
M_size = select_number * 6
N_size = 11

LabM_size = select_number * 6
LabN_size = 6

G_size = select_number * 6
Zn_size = 100

lr_g = 0.0001
lr_D = 0.0001

train_epoch = args.ite

# onehot = np.eye(20)
# variables : input
with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, N_size))
    y = tf.placeholder(tf.float32, shape=(None, LabN_size))
    z = tf.placeholder(tf.float32, shape=(None, Zn_size))
    gy = tf.placeholder(tf.float32, shape=(None, LabN_size))
    isTrain = tf.placeholder(dtype=tf.bool)


# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


# G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        cat1 = tf.concat([x, y], 1)
        z = slim.fully_connected(cat1, 128, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 64, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 32, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 11, activation_fn=tf.nn.relu)

        return z


# D(x)
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        cat1 = tf.concat([x, y], 1)

        x = slim.fully_connected(cat1, 64, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 64, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 32, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 1, activation_fn=None)

        return x


def one_hot(y, size, reuse=False):
    label = []
    for i in range(size):
        a = int(y[i]) - 1
        temp = [0, 0, 0, 0, 0, 0]
        temp[a] = 1
        if i == 0:
            label = temp
        else:
            label.extend(temp)
    label = np.array(label).reshape(size, 6)
    return label


def G_labels(select_num, size):
    temp_z_ = np.random.uniform(-1, 1, (select_num, size))
    z_ = temp_z_
    fixed_y_ = np.ones((select_num, 1))
    j = 1
    for i in range(5):
        temp = np.ones((select_num, 1)) + j
        fixed_y_ = np.concatenate([fixed_y_, temp], 0)
        j = j + 1
        z_ = np.concatenate([z_, temp_z_], 0)  # 矩阵拼接[10*100]*10
    y = one_hot(fixed_y_, M_size, reuse=True)
    return y, z_


def Generate_date():
    temp_z_ = np.random.uniform(-1, 1, (1500, 100))
    z_ = temp_z_
    fixed_y_ = np.ones((1500, 1))
    j = 1
    for i in range(5):
        temp = np.ones((1500, 1)) + j
        fixed_y_ = np.concatenate([fixed_y_, temp], 0)
        j = j + 1
        z_ = np.concatenate([z_, temp_z_], 0)  # 矩阵拼接[10*100]*10
    y = one_hot(fixed_y_, 9000, reuse=True)
    name = "labels1500" + ".txt"
    np.savetxt(name, fixed_y_)
    return y, z_


def show_result(epoch_num, reuse=False):
    with tf.variable_scope('show_result', reuse=reuse):
        if epoch_num == train_epoch-1:
            G_y, fixed_z_ = Generate_date()
            G = sess.run(G_z, {z: fixed_z_, gy: G_y, isTrain: True})
            G_sample = G
            #            name = str(epoch_num) + ".txt"
            name = "GAN_data_base_" + str(select_number) + ".txt"
            np.savetxt(name, G_sample)
            return G_sample

# networks : generator

G_z = generator(z, gy, isTrain)

# Wgan trick

eps = tf.random_uniform(shape=[G_size, 1], minval=0., maxval=1.)
X_inter = eps * x + (1. - eps) * G_z
grad = tf.gradients(discriminator(X_inter, y, isTrain), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2, axis=1))
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))

# networks : discriminator
D_real_logits = discriminator(x, y, isTrain, reuse=True)
D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)

D_loss = tf.reduce_mean(D_fake_logits) - tf.reduce_mean(D_real_logits) + grad_pen
G_loss = -tf.reduce_mean(D_fake_logits)
# trainable variables for each network

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr_D, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr_g, beta1=0.5).minimize(G_loss, var_list=G_vars)

# results save folder
root = 'data_results/'
model = 'data_cGAN_'


# open session and initialize all variables
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=gpuConfig)
sess.run(tf.global_variables_initializer())

# training-loop

print('training start!')
x_, y_ = load_data(select_number, M_size, reuse=True)
for epoch in range(train_epoch):
    epoch_start_time = time.time()
    # upadta Discriminator

    labels = one_hot(y_, M_size, reuse=True)
    z_ = np.random.uniform(-1, 1, (G_size, Zn_size))

    D_losses, _ = sess.run([D_loss, D_optim], {x: x_, y: labels, z: z_, gy: labels, isTrain: True})

    # updata generator

    z_ = np.random.uniform(-1, 1, (G_size, Zn_size))
    G_y, _ = G_labels(select_number, Zn_size)
    G_losses, _, _ = sess.run([G_loss, G_z, G_optim], {x: x_, y: labels, z: z_, gy: G_y, isTrain: True})

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    G = show_result(epoch)
