import tensorflow as tf
import numpy as np

#Declare list of features. Only real-valued features
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

#An estimator is the front end to invoke training(fitting) and evaluation (inference)
#Following code provides an estimetor that does linear regression
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1, 2, 3, 4])
y = np.array([0, -1, -2, -3])
#'numpy_input_fn' is a helper method to read and set-up data sets.
#We have to tell the function how many batches of data (num_epoch) we want and how
#big the estimation should be.
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)
#We can invoke 1000 training steps by invoking fit method and passing the trainig data set.
estimator.fit(input_fn=input_fn, steps=1000.)
print(estimator.evaluate(input_fn=input_fn))

#Custom Model
print("--- Custom model ---")
#If we want to create a model not builded in tensorflow, we can retain the high
#level abstraction  of data-set, training, etc of tf.contrib.learn
#Let's implement custom LinearRegressor

#Per definire un modello che lavor con tf.contrib.learn dobbiamo usare
#tf.contrib.learn.Estimator.
#We provide Estimator a function model_fn that tells tf.contrib.learn how it can
#evaluate predictions, training_steps, ad loss.

def model(features, labels, mode):
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W*features['x']+b
    loss = tf.reduce_sum(tf.square(y-labels))
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss= loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x=np.array([1., 2., 3., 4.])
y=np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)
# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
