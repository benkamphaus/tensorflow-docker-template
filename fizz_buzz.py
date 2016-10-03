# Fizz Buzz in Tensorflow!
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
# this version from: http://benkampha.us/2016-05-24.html
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.set_random_seed(42)

NUM_DIGITS = 10

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])

# We'll want to randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, seed=42))

# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU
# activation. The softmax (which turns arbitrary real-valued outputs into
# probabilities) gets applied in the cost function.

def model(X, w_h1, w_h2, w_o, keep_prob):
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, w_h1)), keep_prob, seed=42)
    h2 = tf.nn.relu(tf.matmul(h1, w_h2))
    return tf.matmul(h2, w_o)

# Our var2iables. The input has width NUM_DIGITS, and the output has width 4.
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])
keep_prob = tf.placeholder(tf.float32)

# How many units in the hidden layer.
NUM_HIDDEN1 = 1250
NUM_HIDDEN2 = 750

# Initialize the weights.
w_h1 = init_weights([NUM_DIGITS, NUM_HIDDEN1])
w_h2 = init_weights([NUM_HIDDEN1, NUM_HIDDEN2])
w_o = init_weights([NUM_HIDDEN2, 4])

# Predict y given x using the model.
py_x = model(X, w_h1, w_h2, w_o, keep_prob)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.0003, decay=0.8, momentum=0.4).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

# Finally, we need a way to turn a prediction (and an original number)
# into a fizz buzz output
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

BATCH_SIZE = 64
numbers = np.arange(1, 101)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(125):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of BATCH_SIZE inputs.
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end],
                                          Y: trY[start:end],
                                          keep_prob: 0.5})

        print(epoch, np.mean(np.argmax(trY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: trX,
                                                             Y: trY,
                                                             keep_prob: 1.0})))

    # And now for some fizz buzz
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
    teY = sess.run(predict_op, feed_dict={X: teX,
                                          keep_prob: 1.0})
    output = np.vectorize(fizz_buzz)(numbers, teY)
    secret = np.vectorize(fizz_buzz)(numbers, [np.argmax(fizz_buzz_encode(n)) for n in numbers])
    acc = np.sum(output == secret)/float(len(output))
    print(output)
    print(acc)
