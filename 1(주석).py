import tensorflow as tf
import pandas as pd
import numpy as np

# 학습 데이터 (입력값과 정답값)
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# 가중치(W)와 편향(b) 변수 선언 (랜덤 초기화)
w = tf.Variable(tf.random.normal([1]), name='weight')  # w: 가중치
b = tf.Variable(tf.random.normal([1]), name='bias')    # b: 편향

# 가설(예측값): y = wx + b
hypothesis = x_train * w + b

# 비용 함수(손실 함수): MSE(Mean Squared Error)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# 비용을 최소화하는 최적화 알고리즘 (경사 하강법 사용)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)  # 학습률 0.1
train = optimizer.minimize(cost)  # 비용을 줄이기 위한 학습 단계 정의

# TensorFlow 세션 생성
sess = tf.Session()

# 변수 초기화 (모든 변수 초기화)
sess.run(tf.global_variables_initializer())

# 2000번의 학습 반복 수행
for step in range(2001):

    sess.run(train)  # 최적화 수행

    # 50번마다 현재 학습 상태 출력
    if step % 50 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))  # 현재 step, 비용, W, b 출력
