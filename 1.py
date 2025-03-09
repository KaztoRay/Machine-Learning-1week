import tensorflow as tf
import pandas as pd
import numpy as np

x_train = [1, 2, 3]
y_train = [1, 2, 3]

w = tf.Variable(tf.random_normal[1], 'weight')
b = tf.Variable(tf.random_normal[1], 'bias')

hypothesis = x_train * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize cost (Gradient descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.session()

sess.run(tf.global_variables_initalizer())

for step in range(2001):

    sess.run(train)

    if step % 50 == 0:

        print(step, sess.run(cost), sess.run(w), sess.run(b))