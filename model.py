from __future__ import print_function
import tensorflow as tf


class Network():

    def __init__(self):

        self.num_hidden = 28
        self.time_steps = 28

    def encoder(self, x, code_size):

        # Unstack to get a list of time_steps tensors of shape (batch_size, n_input)
        x = tf.unstack(x, self.time_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = tf.compat.v1.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Dense layers
        y = tf.compat.v1.layers.dense(outputs[-1], 200, tf.nn.relu)
        y = tf.compat.v1.layers.dense(y, 200, tf.nn.relu)
        loc = tf.compat.v1.layers.dense(y, code_size)
        scale = tf.compat.v1.layers.dense(y, code_size, tf.nn.softplus)

        return tf.contrib.distributions.MultivariateNormalDiag(loc, scale)

    def decoder(self, x):

        # Dense layers
        x = tf.compat.v1.layers.dense(x, 24, tf.nn.relu)
        x = tf.compat.v1.layers.dense(x, 7 * 7 * 64, tf.nn.relu)
        x = tf.compat.v1.reshape(x, [-1, 7, 7, 64])

        # Transpose Convolution Layers
        x = tf.compat.v1.layers.conv2d_transpose(x, 64, 3, 2, 'SAME', activation=tf.nn.relu)
        x = tf.compat.v1.layers.conv2d_transpose(x, 32, 3, 2, 'SAME', activation=tf.nn.relu)
        x = tf.compat.v1.layers.conv2d_transpose(x, 1, 3, 1, 'SAME', activation=None)

        img = tf.reshape(x, [-1, 28, 28])

        return tf.contrib.distributions.Independent(tf.contrib.distributions.Bernoulli(img), 2)

    def make_prior(self, code_size):
        mu = tf.zeros(code_size)
        sigma = tf.ones(code_size)
        return tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)

    def plot_samples(self, ax, samples):
        for index, sample in enumerate(samples):
            ax[index].imshow(sample, cmap='gray')
            ax[index].axis('off')

    def optimization(self, x, learning_rate):

        encode = tf.make_template('encoder', self.encoder)
        decode = tf.make_template('decoder', self.decoder)

        # Define the model
        prior = self.make_prior(code_size=2)
        posterior = encode(x, code_size=2)
        code = posterior.sample()

        # Define the loss
        likelihood = decode(code).log_prob(x)
        divergence = tf.contrib.distributions.kl_divergence(posterior, prior)
        elbo = tf.reduce_mean(likelihood - divergence)
        optimize = tf.train.AdamOptimizer(learning_rate).minimize(-elbo)

        reg = divergence
        samples = decode(prior.sample(10)).mean()

        return optimize, samples, elbo, reg, code