from __future__ import absolute_import, division, print_function
from context import src
import tensorflow.contrib.eager as tfe


import tensorflow as tf

tf.enable_eager_execution()




NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def prediction( input, weight, bias):
    return input * weight + bias

def loss(weights, biases):
    error = prediction( training_inputs , weights, biases) - training_outputs
    return tf.reduce_mean(tf.square(error))

def grad(weights,biases):
    with tfe.GradientTape() as tape:
        loss_value = loss(weights, biases)
    return tape.gradient(loss_value, [ weights, biases])

train_steps = 200
learning_rate = 0.01
W = tfe.Variable(5.)
B = tfe.Variable(10.)

print("Initial loss : {:.3f}".format(loss(W,B)))

for i in range(train_steps):
    dW,dB = grad( W, B)
    W.assign_sub(dW * learning_rate)
    B.assign_sub(dB * learning_rate)
    if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i,loss(W,B)))

print("Final loss: {:.3f}".format(loss(W,B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))






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
