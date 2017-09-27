from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request as ur

import numpy as np
import tensorflow as tf

# Data sets
 

def main():
  
  
  # Load datasets.
  _w = np.random.randint(10,20);
  _b = np.random.randint(10,20);
  
  
  _x = np.floor(10 * np.random.random([5]),dtype=np.float32)
  _y = _x * _w + _b; 
  
  print ("x:", _x)
  print ("y:", _y)  
  
  # Start
  x = tf.placeholder(tf.float32)
  y = tf.placeholder(tf.float32)
  weights = tf.Variable(0.0)
  biases = tf.Variable(0.0)
  y_pre = x * weights + biases 
  
  loss = tf.reduce_sum(tf.square(y_pre - y))             #损失函数，实际输出数据和训练输出数据的方差
  optimizer = tf.train.GradientDescentOptimizer(0.001)
  train = optimizer.minimize(loss)                        #训练的结果是使得损失函数最小

  sess = tf.Session()                                     #创建 Session
  sess.run(tf.global_variables_initializer())             #变量初始化

  for i in range(10000):
    sess.run(train, {x:_x, y:_y})
    if i % 100 == 0:
      print (i, "=", sess.run([weights, biases, loss],{x:_x, y:_y}))
  print ("x=", _w, "b=", _b);
main()