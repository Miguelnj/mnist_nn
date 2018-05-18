import gzip
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)

# ---------------- Visualizing some element of the MNIST dataset --------------

# plt.imshow(train_x[67].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print(train_y[67])

# TODO: the neural net!!

# ----------------- Create the training set --------------

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 25)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(25)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(25, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

# OUTPUT
W3 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b3 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

# "loss" = error
loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
# train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)  # learning rate: 0.005
# train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)  # learning rate: 0.03

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 10
currentLoss = 0
losses = [None]
lossesValid = [None]
previousLoss = 10
epochToPrint = 0

for epoch in range(99999):
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    currentLoss = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})/batch_size
    currentLossValid = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})/len(valid_y)
    losses.append(currentLoss)
    lossesValid.append(currentLossValid)

    if abs(currentLossValid - previousLoss) < 0.003:
        break
    previousLoss = currentLossValid

    print("Epoch:", epoch, "Error: ", currentLoss)
    print("ErrorVal: ", currentLossValid)

    # result = sess.run(y, feed_dict={x: batch_xs})
    # for b, r in zip(batch_ys, result):
    #    print(b, "-->", r)
    # print("----------------------------------------------------------------------------------")
    epochToPrint = epoch
print("/****************** TESTING ******************/")

result = sess.run(y, feed_dict={x: test_x})

rightGuesses = 0
failGuesses = 0

for b, r in zip(test_y, result):
    if np.argmax(b) == np.argmax(r):
        rightGuesses += 1
    else:
        failGuesses += 1
    print(b, "-->", r)
    print("Guesses: ", rightGuesses)
    print("Fails: ", failGuesses)
    tries = rightGuesses + failGuesses
    print("Percentage of right guesses: ", (float(rightGuesses) / float(tries)) * 100, "%")
print("----------------------------------------------------------------------------------")

print(epochToPrint)
print(losses[-1])
print(lossesValid[-1])

plt.plot(losses)
plt.plot(lossesValid)
plt.show()

# 25 neurons epoch = 8 error = 0.0165 errorValid = 0.1112 l.rate = 0.01 guesses = 92.0%
# 20 neurons epoch = 10 error = 0.0256 errorValid = 0.1094 l.rate = 0.01 guesses: 91.9%
# 15 neurons epoch = 10 error = 0.0470 errorValid = 0.1337 l.rate = 0.01 guesses: 90.7%
# 10 neurons epoch = 6 error = 0.0283 errorValid = 0.1634 l.rate = 0.01 guesses: 89.13%

# 25 neurons gives less guesses.
# 20 neurons epoch = 3 error = 0.0137 errorValid = 0.1174 l.rate = 0.03 guesses: 91.94%
# 15 neurons epoch = 2 error = 0.0138 errorValid = 0.1439 l.rate = 0.03 guesses: 89.92%
# 10 neurons epoch = 5 error = 0.06737 errorValid = 0.1585 l.rate = 0.03 guesses: 89.58%

# 20 neurons epoch = 13 error = 0.0869 errorValid = 0.1272 l.rate = 0.005 guesses: 91.24%
# 15 neurons epoch = 12 error = 0.0264 errorValid = 0.1469 l.rate = 0.005 guesses: 89.84%
