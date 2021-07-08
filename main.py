import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import model
# tf version 1.14.0


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


# Parameters
learning_rate = 0.01
num_epochs = 25
last_improvement = 0
require_improvement = 3
best_elbo = 1000

# Build model, Define class
nn = model.Network()
data = tf.placeholder(tf.float32, [None, 28, 28])

optimize, samples, elbo, reg, code = nn.optimization(data, learning_rate)

# Load Data, Create Base Figure for Visualization
mnist = input_data.read_data_sets('MNIST_data/')
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

loss_history = []
reg_history = []

with tf.train.MonitoredSession() as sess:
  sess.run(init_op)

  for epoch in range(num_epochs):
    feed = {data: mnist.test.images.reshape([-1, 28, 28])}
    test_elbo, test_reg, test_codes, test_samples = sess.run([elbo, reg, code, samples], feed)
    loss_history.append(-1 * test_elbo)
    reg_history.append(test_reg.mean())
    print('Epoch', epoch, '- elbo', -1 * test_elbo, '- divergence', test_reg.mean())
    if -1 * test_elbo >= best_elbo:
        last_improvement = last_improvement + 1
    else:
        best_elbo = -1 * test_elbo

    if last_improvement > require_improvement:
        break

    for _ in range(600):
      feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])}
      sess.run(optimize, feed)

  saver.save(get_session(sess), 'new_model.ckpt')

  for i in range(0, 10):
      feed = {data: mnist.test.images.reshape([-1, 28, 28])}
      test_elbo, test_reg, test_codes, test_samples = sess.run([elbo, reg, code, samples], feed)
      nn.plot_samples(ax[i, :], test_samples)

plt.savefig('new_images.png', dpi=300, transparent=True, bbox_inches='tight')