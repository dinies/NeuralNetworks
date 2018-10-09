# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    with tf.name_scope('total'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          labels=y_, logits=y)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)














# from __future__ import absolute_import, division, print_function

# from context import src
# import tensorflow.contrib.eager as tfe


# import tensorflow as tf

# tf.enable_eager_execution()




# NUM_EXAMPLES = 1000
# training_inputs = tf.random_normal([NUM_EXAMPLES])
# noise = tf.random_normal([NUM_EXAMPLES])
# training_outputs = training_inputs * 3 + 2 + noise

# def prediction( input, weight, bias):
#     return input * weight + bias

# def loss(weights, biases):
#     error = prediction( training_inputs , weights, biases) - training_outputs
#     return tf.reduce_mean(tf.square(error))

# def grad(weights,biases):
#     with tfe.GradientTape() as tape:
#         loss_value = loss(weights, biases)
#     return tape.gradient(loss_value, [ weights, biases])

# train_steps = 200
# learning_rate = 0.01
# W = tfe.Variable(5.)
# B = tfe.Variable(10.)

# print("Initial loss : {:.3f}".format(loss(W,B)))

# for i in range(train_steps):
#     dW,dB = grad( W, B)
#     W.assign_sub(dW * learning_rate)
#     B.assign_sub(dB * learning_rate)
#     if i % 20 == 0:
#         print("Loss at step {:03d}: {:.3f}".format(i,loss(W,B)))

# print("Final loss: {:.3f}".format(loss(W,B)))
# print("W = {}, B = {}".format(W.numpy(), B.numpy()))






# w = tfe.Variable([[1,0]])
# with tfe.GradientTape() as tape:
#     loss = w * w

# grad = tape.gradient(loss, [w,w,w])
# print(loss)
# print(grad)









# P = Perceptron()
# P.train()
# P.test()
# P.deployBoard()
# P.save


# #create 2 floating point Tensors node1 and node2
# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)
# print(node1, node2)
# #The print does not output the value 3.0 and 4.0
# #They are nodes that, when evaluated, would produce 3.0 and 4.0 respectively.

# #To evaluate the nodes, we must run the graph in a session.
# sess = tf.Session()
# print(sess.run([node1, node2]))

# node3 = tf.add(node1, node2) #addition
# print("node3: ", node3)
# print("sess.run(node3): ", sess.run(node3))

# #A graph can be parametrized  to accept external input, known as placeholder.
# #A placeholder is a promise to provide a value later.
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b #provides a shorcut for tf.add

# print(sess.run(adder_node, {a:3 , b:4.5}))
# print(sess.run(adder_node, {a:[1,3], b: [2,4]}))

# add_and_triple = adder_node*3
# print(sess.run(add_and_triple, {a: 3, b:4.5}))


# W = tf.Variable([0.3], tf.float32) #Variable allows us to add a trainable parameter to a graph
# b = tf.Variable([-0.3], tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W*x+b
# #tf.constant initialize a Variable, whose value can never change.
# #By constrast, variables are not initialized when you call tf.Variable.
# #To initialize all the variables you must explicitly call a special operation:
# init = tf.global_variables_initializer()
# sess.run(init) #untill we call sess.run the variables are unitialized
# print(sess.run(linear_model, {x:[1,2,3,4]}))

# y=tf.placeholder(tf.float32)
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
# print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# #A Variable can be changed using operation like tf.assign
# fixW = tf.assign(W, [-1]) #reassign -1 to W
# fixb = tf.assign(b, [1.])
# sess.run([fixW, fixb])
# print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# #Tensorflow provides optimizer, like gradient descent.
# #Tensorflow can automatically produce derivatives given only a description of the model
# #using the function tf.gradients
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# sess.run(init) #init = tf.global_variables_initializer()
# for i in range(1000):
#     sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
#     #train recalls loss, that recalls squared_deltas and linear_model
# print(sess.run([W, b]))

# #More complete script

# # Model parameters
# print('More complete script')
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# y = tf.placeholder(tf.float32)
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# # training data
# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]
# # training loop
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init) # reset values to wrong
# for i in range(1000):
#   sess.run(train, {x:x_train, y:y_train})

# # evaluate training accuracy
# curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
