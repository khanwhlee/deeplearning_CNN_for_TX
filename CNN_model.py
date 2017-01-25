import tensorflow as tf
from data_processing import data_processing
import numpy as np


class CNN_model:

	def __init__(self, n_classes=2, batch_size=5, keep_rate=0.8):
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.keep_rate = keep_rate
		self.keep_prob = tf.placeholder(tf.float32)

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def maxpool2d(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def convolutional_neural_network(self, x):
		weights = {'W_conv1':self.weight_variable([5,5,3,32]),
					'W_conv2':self.weight_variable([3,5,32,64]),
					'W_fc1':self.weight_variable([2*5*64, 1024]),
					'W_fc2':self.weight_variable([1024, 10]),
					'out':self.weight_variable([10, self.n_classes])}
		biases = {'b_conv1':self.bias_variable([32]),
					'b_conv2':self.bias_variable([64]),
					'b_fc1':self.bias_variable([1024]),
					'b_fc2':self.bias_variable([10]),
					'out':self.bias_variable([self.n_classes])}

		x = tf.reshape(x, shape=[-1,8,20,3])
		conv1 = tf.nn.relu(self.conv2d(x, weights['W_conv1']) + biases['b_conv1'])
		conv1 = self.maxpool2d(conv1)

		conv2 = tf.nn.relu(self.conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
		conv2 = self.maxpool2d(conv2)

		fc = tf.reshape(conv2, [-1, 2*5*64])
		fc = tf.nn.relu(tf.matmul(fc, weights['W_fc1']) + biases['b_fc1'])
		fc = tf.nn.relu(tf.matmul(fc, weights['W_fc2']) + biases['b_fc2'])
		fc = tf.nn.dropout(fc, self.keep_rate)
		output = tf.matmul(fc, weights['out']) + biases['out']

		return output

	def train_neural_network(self,x,y):
		prediction = self.convolutional_neural_network(x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

		optimizer = tf.train.AdamOptimizer().minimize(cost)

		hm_epochs = 3

		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())

			for epoch in range(hm_epochs):
				epoch_loss = 0
				i = 0
				while i < len(train_x):
					start = i
					end = i + self.batch_size
					batch_x = np.array(train_x[start:end])
					batch_y = np.array(train_y[start:end])

					_, c = sess.run([optimizer,cost], feed_dict = {x: batch_x, y: batch_y})
					epoch_loss += c
					i += self.batch_size
				print ('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

			print ('Accuracy:' ,accuracy.eval({x:test_x, y:test_y}))


train_x, train_y, test_x, test_y = data_processing()
x = tf.placeholder('float',[None, 8, 20, 3])
y_label = tf.placeholder('float')
test_cnn = CNN_model()
test_cnn.train_neural_network(x,y_label)

