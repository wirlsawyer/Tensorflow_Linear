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
  test_count = 10         # row count
  param_count = 5         # x1, x2, x3, x4, x5
  
  _x = np.floor(10 * np.random.random([test_count, param_count]),dtype=np.float32)
  _w = np.floor(10 * np.random.random([param_count, 1]),dtype=np.float32)
  _b = np.floor(10 * np.random.random([test_count, 1]), dtype=np.float32)  
  _y = _x.dot(_w) #+ _b
  
  print ("X:", _x)
  print ("W:", _w)
  print ("B:", _b)
  print ("Y:", _y)

  # Start
  x = tf.placeholder(tf.float32,shape=[test_count, param_count])
  y = tf.placeholder(tf.float32,shape=[test_count, 1])
  weights = tf.Variable(np.zeros(param_count, dtype=np.float32).reshape((param_count,1)), tf.float32)
  biases = tf.Variable(tf.zeros([test_count, 1]), dtype=np.float32)
    
  y_pre = tf.matmul(x, weights)               
  #y_pre = tf.add(tf.matmul(x, weights), biases)          
  
  
  loss = tf.reduce_sum(tf.square(y_pre -y))   
  optimizer = tf.train.GradientDescentOptimizer(0.001)
  train = optimizer.minimize(loss)                 
 
  # Terminal value
  LOSS_MIN_VALUE = tf.constant(1e-1)               

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  run_count = 0
  last_loss = 0
  while True:
    run_count = run_count + 1
    sess.run(train, {x:_x, y:_y})

    curr_loss,is_ok = sess.run([loss,loss < LOSS_MIN_VALUE],{x:_x, y:_y})
    
    if run_count % 50 == 0:
      print (run_count, " loss=", curr_loss)

    if last_loss == curr_loss:
      print (run_count, "final loss=", curr_loss)
      break

    last_loss = curr_loss
    if is_ok:
      print (run_count, "final loss=", curr_loss)
      break

  curr_W, curr_loss, curr_biases = sess.run([weights, loss, biases], {x:_x, y:_y})
  print("####################################################")
  print ("W:", _w)
  print ("B:", _b)
  print("####################################################")
  print("weights: %s" % (weights) )
  print("fix_w: %s" % (np.round(curr_W)) )
  print("####################################################")
  print("biases: %s" % (biases) )
  print("fix_biases: %s" % (np.round(curr_biases)) )
  print("####################################################")
  
  print("loss: %s" % (curr_loss) )
  print("fix_loss: %s" % ( np.sum(np.square(weights - np.round(curr_W)))  ) )
   
main()