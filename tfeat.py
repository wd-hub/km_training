import tensorflow as tf
import numpy as np

class TFeat(object):
    def __init__(self, num_channels, tfeat_npy_path=None):
        self.num_channels = num_channels

        if tfeat_npy_path is not None:
            self.data_dict = np.load(tfeat_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

    def build(self, data, is_training=None, reuse=False):
        params = dict()   # use for intermediate analysis
        self.conv1 = self.conv_layer(data, self.num_channels,32,7,1, 'VALID', 'conv1', reuse)
        self.bn1   = self.bn(self.conv1, 'bn1', reuse, is_training)
        self.tanh1 = tf.nn.tanh(self.bn1)
        self.pool1 = tf.nn.max_pool(self.tanh1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        self.conv2 = self.conv_layer(self.pool1, 32,64,6,1, 'VALID', 'conv2', reuse)
        self.bn2   = self.bn(self.conv2, 'bn2', reuse, is_training)
        self.tanh2 = tf.nn.tanh(self.bn2)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        tanh_shape = self.tanh2.get_shape().as_list()
        reshape = tf.reshape(self.tanh2, [-1, 8 * 8 * 64])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        reshape = tf.cond(is_training, lambda: tf.nn.dropout(reshape, 0.9), lambda: reshape)
        self.fn1 = self.fc_layer(reshape, 8*8*64, 128, 'fc1', reuse)
        # batch normalization or not
        self.fn1 = self.bn(self.fn1, 'bn3', reuse, is_training)

        self.norm = tf.sqrt(tf.reduce_sum(tf.multiply(self.fn1, self.fn1), 1, keep_dims=True) + 1e-10)
        self.output = tf.multiply(self.fn1, tf.tile(1. / self.norm, [1, 128]))
        return self.output

    def conv_layer(self, data, in_channels, out_channels, fs, ss, pad, name, reuse):
        # fs: filter size
        # ss: stride size
        with tf.variable_scope(name, reuse=reuse):
            w, b = self.get_conv_var(fs, in_channels, out_channels)
            # if ss == 2:
            #     pattern = [[0, 0],
            #                [1, 0],
            #                [1, 0],
            #                [0, 0]]
            #     data = tf.pad(data, paddings=pattern)
            #     conv = tf.nn.conv2d(data, w, strides=[1, 2, 2, 1], padding='VALID')
            # else:
            conv = tf.nn.conv2d(data, w, [1, ss, ss, 1], padding=pad)
            layer = tf.nn.bias_add(conv, b)
        return layer

    def fc_layer(self, data, in_size, out_size, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            w, b = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(data, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, w), b)
        return fc

    def get_conv_var(self, filter_size, in_channels, out_channels):
        weights = self._variable_with_weight_decay('weights',
                                                    shape=[filter_size, filter_size, in_channels, out_channels],
                                                    wd = 0)
        biases = self._variable_on_device('biases', [out_channels],
                                          tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        return weights, biases

    def get_fc_var(self, in_size, out_size, name):
        weights = self._variable_with_weight_decay('weights',
                                                   shape=[in_size, out_size],
                                                   wd = 0)
        biases = self._variable_on_device('biases',[out_size],
                                          tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        return weights, biases

    def _variable_on_device(self, name, shape, initializer):
        """Helper to create a Variable stored on GPU memory.
        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
        Returns:
            Variable Tensor
        """
        with tf.device('cpu:0'):
            dtype = tf.float32
            layer = tf.get_variable_scope().name
            if self.data_dict is not None and layer in self.data_dict:
                value = self.data_dict[layer][name]
                var = tf.get_variable(name, initializer=value, dtype=dtype)
            else:
                value = initializer
                var = tf.get_variable(name, shape, initializer=value, dtype=dtype)
        return var

    def _variable_with_weight_decay(self, name, shape, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        var = self._variable_on_device(
            name,
            shape,
            tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

        # We will replicate the model structure for the training subgraph, as well
        # as the evaluation subgraphs, while sharing the trainable parameters.

    def bn(self, data, name, reuse, is_training):
        batchNorm = tf.contrib.layers.batch_norm(data, decay=0.9, center=True, scale=True,
                                            updates_collections=None,
                                            is_training=is_training,
                                            reuse=reuse,
                                            trainable=True,
                                            scope=name)
        # batchNorm = tf.nn.batch_normalization(data, self.data_dict[name]['running_mean'],
        #                                 self.data_dict[name]['running_var'], 0, 1, 1e-5)
        return batchNorm