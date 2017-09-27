import os
import numpy as np
import tensorflow as tf
import pandas as pd
 
#Read File
ALL_DATA = pd.read_csv('D:\Python\Tensorflow_Linear\data.csv', header=0) 

#define data
csv_data1 = ALL_DATA.data1.as_matrix().reshape(len(ALL_DATA),1)
csv_data2 = ALL_DATA.data2.as_matrix().reshape(len(ALL_DATA),1)
csv_ans = ALL_DATA.ans.as_matrix().reshape(len(ALL_DATA),1)

x_data = np.zeros((len(ALL_DATA), 2))
x_data = np.concatenate((csv_data1, csv_data2), axis = 1)

y_data = np.zeros((len(ALL_DATA), 1))
y_data = csv_ans

x_dimension = len(x_data[0])  
y_dimension = len(y_data[0]) 
rowCount = len(x_data)

def main():
   
  _x = x_data;
  _y = y_data;
  
  # Start
  x = tf.placeholder(tf.float32,shape=[rowCount, x_dimension])
  y = tf.placeholder(tf.float32,shape=[rowCount, 1])
  weights = tf.Variable(np.zeros(x_dimension, dtype=np.float32).reshape((x_dimension,1)), tf.float32)
  biases = tf.Variable(tf.zeros([rowCount, 1]), dtype=np.float32)
    
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
  print("weights: %s" % sess.run(weights) )
  print("fix_w: %s" % (np.round(curr_W)) )
  print("####################################################")
  print("biases: %s" % sess.run(biases) )
  print("fix_biases: %s" % (np.round(curr_biases)) )
  print("####################################################")
  
  print("loss: %s" % (curr_loss) )
  print("fix_loss: %s" % sess.run( np.sum(np.square(weights - np.round(curr_W)))  ) )
   
main()