import tensorflow as tf
import os

# dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [0], [0], [1]]

# model param
N_INPUT_NODES = 2
N_OUTPUT_NODES = 1

# train times
N_STEPS = 2000
N_EPOCH = 100

LEARNING_RATE = 0.02

x_ = tf.placeholder(tf.float32, shape=[len(X), N_INPUT_NODES], name="x-input")    # 4 x 2 matrics
y_ = tf.placeholder(tf.float32, shape=[len(Y), N_OUTPUT_NODES], name="y-output")  # 4 x 1 matrics

weight = tf.Variable(tf.random_uniform([N_INPUT_NODES, N_OUTPUT_NODES], -1, 1), name="weight") # 2 x 1 matrics
bias = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias")                                    # 1 x 1 matrics

# forward
output = tf.sigmoid(tf.matmul(x_, weight) + bias)

# loss
cost = tf.reduce_mean(tf.square(Y - output))

# back gradient descent
train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

# init var
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# start train
for i in range(N_STEPS):
    # execute train func, feed dataset to model
    sess.run(train, feed_dict={x_: X, y_: Y})
    if i % N_EPOCH == 0:
        print('STEPS: ', i, ' cost: ', sess.run(cost, feed_dict={x_: X, y_: Y}))

print('output: ', sess.run(output, feed_dict={x_: X, y_: Y}))

tensorboard_dir = './tensorboard/tf_grad_and'

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

writer = tf.summary.FileWriter(tensorboard_dir)
writer.add_graph(sess.graph)