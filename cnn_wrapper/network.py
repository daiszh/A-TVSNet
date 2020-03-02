#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from six import string_types
slim = tf.contrib.slim

# Zero padding in default. 'VALID' gives no padding.
DEFAULT_PADDING = 'SAME'

def layer(op):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        """Layer decoration."""
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if not self.terminals:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    """Class NetWork"""

    def __init__(self, inputs, is_training, dropout_rate=0.9, seed=None, reuse=False, scope_name=None):
        # initializer for conv and deconv
        self.kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None)
        # self.kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-4, seed=None)
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = is_training
        # If true, variables are shared between feature towers
        self.reuse = reuse
        self.scope_name = scope_name # only used in manke_var() when reuse==False
        # If true, layers like batch normalization or dropout are working in training mode
        self.training = is_training
        # Seed for randomness
        self.seed = seed
        # Dropout rate
        self.dropout_rate = dropout_rate
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert args
        self.terminals = []
        for fed_layer in args:
            # if isinstance(fed_layer, basestring):
            if isinstance(fed_layer, string_types):
                # print('fed_layer', fed_layer, self.layers[fed_layer])
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            elif isinstance(fed_layer, list):
                if len(fed_layer)==2 and isinstance(fed_layer[0], Network) and isinstance(fed_layer[1], string_types):
                    # print('fed_layer', fed_layer[1], fed_layer[0])
                    try:
                        fed_layer = fed_layer[0].get_output_by_name(fed_layer[1])
                    except KeyError:
                        raise KeyError('Unknown layer name fed: %s' % fed_layer[1])

            self.terminals.append(fed_layer) 
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_output_by_name(self, layer_name):
        '''
        Get graph node by layer name
        :param layer_name: layer name string
        :return: tf node
        '''
        return self.layers[layer_name]

    def get_shape_by_name(self, layer_name):
        '''
        Get shape of graph node by layer name
        :param layer_name: layer name string
        :return: tf shape
        '''
        return tf.shape(self.layers[layer_name])

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def change_inputs(self, inputs):
        assert len(inputs) == 1
        for key in inputs:
            self.layers[key] = inputs[key]

    @layer
    def conv(self,
             input,
             kernel_size,
             filters,
             strides,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             biased=False,
             rate=1):
        kwargs = {'filters': filters,
                  'kernel_size': kernel_size,
                  'strides': strides,
                  'activation': tf.nn.relu if relu else None,
                  'use_bias': biased,
                  'padding': padding,
                  'trainable': self.trainable,
                  'reuse': self.reuse,
                  'name': name,
                  'kernel_initializer': self.kernel_initializer,
                  'dilation_rate': rate
                  }
        if len(input.get_shape()) == 4:
            return tf.layers.conv2d(input, **kwargs)
        elif len(input.get_shape()) == 5:
            return tf.layers.conv3d(input, **kwargs)
        else:
            raise ValueError('Improper input rank for layer: ' + name)


    @layer
    def conv_bn(self,
                input,
                kernel_size,
                filters,
                strides,
                name,
                relu=True,
                center=False,
                padding=DEFAULT_PADDING,
                biased=False,
                rate=1):
        kwargs = {'filters': filters,
                'kernel_size': kernel_size,
                'strides': strides,
                'activation': None,
                'use_bias': biased,
                'padding': padding,
                'trainable': self.trainable,
                'reuse': self.reuse,
                'kernel_initializer': self.kernel_initializer,
                'dilation_rate': rate
                }

        with tf.variable_scope(name):
            if len(input.get_shape()) == 4:
                conv = tf.layers.conv2d(input, **kwargs)
            elif len(input.get_shape()) == 5:
                conv = tf.layers.conv3d(input, **kwargs)
            else:
                raise ValueError('Improper input rank for layer: ' + name)

            # note that offset is disabled in default
            # scale is typically unnecessary if next layer is relu.
            output = tf.layers.batch_normalization(conv,
                                                   center=center,
                                                   scale=False,
                                                   training=self.training,
                                                   fused=True,
                                                   trainable=center,
                                                   reuse=self.reuse)
            if relu:
                output = tf.nn.relu(output)
            return output


    @layer
    def split_separable_conv2d(self,
                            inputs,
                            kernel_size,
                            filters,
                            rate,
                            name,
                            weight_decay=0.00004,
                            depthwise_weights_initializer_stddev=0.33,
                            pointwise_weights_initializer_stddev=0.06):
        """Splits a separable conv2d into depthwise and pointwise conv2d.

        This operation differs from `tf.layers.separable_conv2d` as this operation
        applies activation function between depthwise and pointwise conv2d.

        Args:
            inputs: Input tensor with shape [batch, height, width, channels].
            filters: Number of filters in the 1x1 pointwise convolution.
            kernel_size: A list of length 2: [kernel_height, kernel_width] of
            of the filters. Can be an int if both values are the same.
            rate: Atrous convolution rate for the depthwise convolution.
            weight_decay: The weight decay to use for regularizing the model.
            depthwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for depthwise convolution.
            pointwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for pointwise convolution.
            scope: Optional scope for the operation.

        Returns:
            Computed features after split separable conv2d.
        """
        outputs = slim.separable_conv2d(
            inputs,
            None,
            kernel_size=kernel_size,
            depth_multiplier=1,
            rate=rate,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=depthwise_weights_initializer_stddev),
            weights_regularizer=None,
            scope=name + '_depthwise',
            trainable=self.trainable)
        return slim.conv2d(
            outputs,
            filters,
            1,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=pointwise_weights_initializer_stddev),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            scope=name + '_pointwise',
            trainable=self.trainable)


    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    def make_var(self, name, shape, initializer=None):
        '''Creates a new TensorFlow variable.'''
        # if self.reuse==False:
        #     name = self.scope_name + '_' + name
        return tf.get_variable(name, shape, initializer=self.kernel_initializer if initializer is None else initializer, trainable=self.trainable)

    # @layer
    def attention_activation(self,
                            input,
                            kernel_size,
                            name,
                            filters=None,
                            second_weight=False,
                            relu=True,
                            padding=DEFAULT_PADDING,
                            biased=False,
                            n_view = None):
        ''' input size (B, D, H, W, C, N) -> share weight 3D conv -> 
            output size (B, D, H, W, C, N) '''
        shape_i = tf.shape(input)
        c_in = input.get_shape().as_list()[-2]
        if n_view is None:
            n_view = input.get_shape().as_list()[-1]
        if filters is None:
            filters = c_in

        def conv(old,elems):
            x_sample = elems[0]
            kernel_sample = elems[1]
            y_sample = tf.nn.conv3d(x_sample,kernel_sample,strides=[1,1,1,1,1],padding=padding)
            if biased:
                biases = self.make_var(name='biases', shape=[y_sample.get_shape()[-1]], initializer=tf.zeros_initializer())
                y_sample = tf.nn.bias_add(y_sample, biases)
            if relu:
                # ReLU non-linearity
                y_sample = tf.nn.relu(y_sample)
            return y_sample

        with tf.variable_scope(name) as scope:
            weight_unique = self.make_var(name='weight_unique', shape=[kernel_size, kernel_size, kernel_size, c_in, filters])
            weight_unique_scan = tf.tile(tf.expand_dims(weight_unique, axis=0), [n_view,1,1,1,1,1])
            x_scan = tf.transpose(input, [5,0,1,2,3,4])

            if second_weight:
                weight_shared = self.make_var(name='weight_shared', shape=[
                    kernel_size, kernel_size, kernel_size, c_in, filters])
                weight_shared_scan = tf.tile(tf.expand_dims(weight_shared, axis=0), [n_view,1,1,1,1,1])

                output_shared = tf.scan(conv, (x_scan, weight_shared_scan), initializer=tf.zeros(
                    (shape_i[0], shape_i[1], shape_i[2], shape_i[3], filters)))
                # output_shared = tf.transpose(output_shared,[1,2,3,4,5,0])
                output_shared_sum = tf.reduce_sum(output_shared, keepdims=False, axis=0)

                def conv_with_second_weight(old,elems):
                    x_sample = elems[0]
                    kernel_sample = elems[1]
                    y_sample = tf.nn.conv3d(x_sample,kernel_sample,strides=[1,1,1,1,1],padding=padding)
                    if biased:
                        biases = self.make_var(name='biases_with_second_weight', shape=[y_sample.get_shape()[-1]], initializer=tf.zeros_initializer())
                        y_sample = tf.nn.bias_add(y_sample, biases)
                    if relu:
                        # ReLU non-linearity
                        y_sample = tf.nn.relu(y_sample)
                    y_sample = tf.subtract(y_sample, elems[2])
                    y_sample = tf.add(y_sample, output_shared_sum)
                    return y_sample
                
                output_unique = tf.scan(conv_with_second_weight, (x_scan, weight_unique_scan, output_shared), initializer=tf.zeros(
                    (shape_i[0], shape_i[1], shape_i[2], shape_i[3], filters)))
                output_unique = tf.transpose(output_unique,[1,2,3,4,5,0])

            else:
                output_unique = tf.scan(conv, (x_scan, weight_unique_scan), initializer=tf.zeros(
                    (shape_i[0], shape_i[1], shape_i[2], shape_i[3], filters)))
                output_unique = tf.transpose(output_unique,[1,2,3,4,5,0])
            
            return output_unique


    @layer
    def attention_activation_layer(self,
                                   input,
                                   kernel_size,
                                   name,
                                   filters=None,
                                   second_weight=False,
                                   relu=True,
                                   padding=DEFAULT_PADDING,
                                   biased=False,
                                   n_view=None):
        with tf.variable_scope(name) as scope:
            activated = self.attention_activation(input=input,
                                                kernel_size=kernel_size,
                                                name=name,
                                                filters=filters,
                                                second_weight=second_weight,
                                                relu=relu,
                                                padding=padding,
                                                biased=biased,
                                                n_view=n_view)
            score_map = tf.nn.softmax(activated, axis=-1, name='att_softmax')
            return score_map

    @layer
    def attention_aggregation(self,
                              input,
                              kernel_size,
                              name,
                              filters=None,
                              second_weight=False,
                              relu=True,
                              padding=DEFAULT_PADDING,
                              biased=False,
                              n_view = None):
        # input (B, D, H, W, C, N)
        # output (B, D, H, W, C)
        with tf.variable_scope(name) as scope:
            activated = self.attention_activation(input=input,
                                                kernel_size=kernel_size,
                                                name='attention_activation',
                                                filters=filters,
                                                second_weight=second_weight,
                                                relu=relu,
                                                padding=padding,
                                                biased=biased,
                                                n_view=n_view)

            score_map = tf.nn.softmax(activated, axis=-1, name='att_softmax')

            aggregated = tf.reduce_sum(tf.multiply(
                score_map, input, name='weighted'), 
                axis=-1, name='aggregated', keepdims=False)

            return aggregated


    @layer
    def attention_activation_2d(self,
                            input,
                            kernel_size,
                            name,
                            filters=None,
                            second_weight=False,
                            relu=True,
                            padding=DEFAULT_PADDING,
                            biased=False):
        ''' input size (B, H, W, C, N) -> share weight 2D conv -> 
            output size (B, H, W, C, N) '''
        shape_i = tf.shape(input)
        c_in = input.get_shape().as_list()[-2]
        n_view = input.get_shape().as_list()[-1]
        if filters is None:
            filters = c_in

        def conv(old,elems):
            x_sample = elems[0]
            kernel_sample = elems[1]
            y_sample = tf.nn.conv2d(x_sample,kernel_sample,strides=[1,1,1,1],padding=padding)
            if biased:
                biases = self.make_var(name='biases', shape=[y_sample.get_shape()[-1]], initializer=tf.zeros_initializer())
                y_sample = tf.nn.bias_add(y_sample, biases)
            if relu:
                # ReLU non-linearity
                y_sample = tf.nn.relu(y_sample)
            return y_sample

        with tf.variable_scope(name) as scope:
            weight_unique = self.make_var(name='weight_unique', shape=[kernel_size, kernel_size, c_in, filters])
            weight_unique_scan = tf.tile(tf.expand_dims(weight_unique, axis=0), [n_view,1,1,1,1])
            x_scan = tf.transpose(input, [4,0,1,2,3])

            if second_weight:
                weight_shared = self.make_var(name='weight_shared', shape=[kernel_size, kernel_size, c_in, filters])
                weight_shared_scan = tf.tile(tf.expand_dims(weight_shared, axis=0), [n_view,1,1,1,1])

                output_shared = tf.scan(conv, (x_scan, weight_shared_scan), initializer=tf.zeros(
                    (shape_i[0], shape_i[1], shape_i[2], filters)))
                # output_shared = tf.transpose(output_shared,[1,2,3,4,0])
                output_shared_sum = tf.reduce_sum(output_shared, keepdims=False, axis=0)

                def conv_with_second_weight(old,elems):
                    x_sample = elems[0]
                    kernel_sample = elems[1]
                    y_sample = tf.nn.conv2d(x_sample,kernel_sample,strides=[1,1,1,1],padding=padding)
                    if biased:
                        biases = self.make_var(name='biases_with_second_weight', shape=[y_sample.get_shape()[-1]], initializer=tf.zeros_initializer())
                        y_sample = tf.nn.bias_add(y_sample, biases)
                    if relu:
                        # ReLU non-linearity
                        y_sample = tf.nn.relu(y_sample)
                    y_sample = tf.subtract(y_sample, elems[2])
                    y_sample = tf.add(y_sample, output_shared_sum)
                    return y_sample
                
                output_unique = tf.scan(conv_with_second_weight, (x_scan, weight_unique_scan, output_shared), initializer=tf.zeros(
                    (shape_i[0], shape_i[1], shape_i[2], filters)))
                output_unique = tf.transpose(output_unique,[1,2,3,4,0])

            else:
                output_unique = tf.scan(conv, (x_scan, weight_unique_scan), initializer=tf.zeros(
                    (shape_i[0], shape_i[1], shape_i[2], filters)))
                output_unique = tf.transpose(output_unique,[1,2,3,4,0])
            
            return output_unique


    @layer
    def deconv(self,
               input,
               kernel_size,
               filters,
               strides,
               name,
               relu=True,
               padding=DEFAULT_PADDING,
               biased=False):
        kwargs = {'filters': filters,
                  'kernel_size': kernel_size,
                  'strides': strides,
                  'activation': tf.nn.relu if relu else None,
                  'use_bias': biased,
                  'padding': padding,
                  'trainable': self.trainable,
                  'reuse': self.reuse,
                  'name': name,
                  'kernel_initializer': self.kernel_initializer
                  }

        if len(input.get_shape()) == 4:
            return tf.layers.conv2d_transpose(input, **kwargs)
        elif len(input.get_shape()) == 5:
            return tf.layers.conv3d_transpose(input, **kwargs)
        else:
            raise ValueError('Improper input rank for layer: ' + name)

    @layer
    def deconv_bn(self,
                  input,
                  kernel_size,
                  filters,
                  strides,
                  name,
                  relu=True,
                  center=False,
                  padding=DEFAULT_PADDING,
                  biased=False):
        kwargs = {'filters': filters,
                  'kernel_size': kernel_size,
                  'strides': strides,
                  'activation': None,
                  'use_bias': biased,
                  'padding': padding,
                  'trainable': self.trainable,
                  'reuse': self.reuse,
                  'kernel_initializer': self.kernel_initializer 
                  }

        with tf.variable_scope(name):
            if len(input.get_shape()) == 4:
                conv = tf.layers.conv2d_transpose(input, **kwargs)
            elif len(input.get_shape()) == 5:
                conv = tf.layers.conv3d_transpose(input, **kwargs)
            else:
                raise ValueError('Improper input rank for layer: ' + name + ', input_shape: ' + str(len(input.get_shape())))
            # note that offset is disabled in default
            # scale is typically unnecessary if next layer is relu.
            output = tf.layers.batch_normalization(conv,
                                                   center=center,
                                                   scale=False,
                                                   training=self.training,
                                                   fused=True,
                                                   trainable=center,
                                                   reuse=self.reuse)
            if relu:
                output = tf.nn.relu(output)
            return output

    def bottleneck(self, inputs, kernel_size, depth, stride=1, rate=1, name=None):
        """Bottleneck residual unit variant with BN before convolutions.
        When putting together two consecutive ResNet blocks that use this unit, one
        should use stride = 2 in the last unit of the first block.
        Args:
        inputs: A tensor of size [batch, height, width, channels].
        depth: The depth of the ResNet unit output.
        depth_bottleneck: The depth of the bottleneck layers.
        stride: The ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        scope: Optional variable_scope.
        Returns:
        The ResNet unit's output.
        """
        scope = name
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]):
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            preact = slim.batch_norm(
                inputs, activation_fn=tf.nn.relu, scope='preact', reuse=self.reuse, trainable=self.training)
            if depth == depth_in:
                if stride == 1:
                    shortcut = inputs
                else:
                    shortcut = slim.max_pool2d(inputs, [1, 1], stride=stride, scope='shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                       normalizer_fn=None, activation_fn=None, reuse=self.reuse, trainable=self.training, weights_initializer=self.kernel_initializer, 
                                       scope='shortcut')

            depth_bottleneck = depth
            residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, reuse=self.reuse, trainable=self.training,
                                   scope='conv1')
            if stride == 1:
                residual = slim.conv2d(residual, depth_bottleneck, kernel_size, stride=1, rate=rate, reuse=self.reuse, trainable=self.training, weights_initializer=self.kernel_initializer, 
                                       padding='SAME', scope='conv2')
            else:
                kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                residual = tf.pad(residual, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
                residual = slim.conv2d(residual, depth_bottleneck, kernel_size, stride=stride, reuse=self.reuse, trainable=self.training, weights_initializer=self.kernel_initializer, 
                                       rate=rate, padding='VALID', scope='conv2')
            
            residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None, reuse=self.reuse, trainable=self.training, weights_initializer=self.kernel_initializer,  
                                   scope='conv3')

            output = shortcut + residual
            return output

    @layer
    def res_block(self, inputs, kernel_size, depth, num_block=1, stride=1, rate=1, name=None):
        if num_block == 1:
            return self.bottleneck(inputs=inputs, kernel_size=kernel_size, depth=depth,
                            stride=stride, rate=rate, name=name)
        else:
            output = self.bottleneck(inputs=inputs, kernel_size=kernel_size, depth=depth, 
                            stride=stride, rate=rate, name=name+'_'+str(0))
            for i in range(1, num_block):
                scope_name = name+'_'+str(i) if i != num_block-1 else name
                output = self.bottleneck(inputs=output, kernel_size=kernel_size, depth=depth, 
                            stride=1, rate=rate, name=scope_name)
            return output
                

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def transpose(self, input, perm, name, conjugate=False):
        return tf.transpose(input, perm=perm, name=name, conjugate=conjugate)

    @layer
    def divide(self, input, denominator, name):
        return tf.divide(input, denominator, name=name)

    @layer
    def reduce_mean(self, input, axis, name, keepdims=True):
        return tf.reduce_mean(input, axis, keepdims=keepdims, name=name)

    @layer
    def reduce_sum(self, input, axis, name, keepdims=True):
        return tf.reduce_sum(input, axis, keepdims=keepdims, name=name)

    @layer
    def tile(self, input, multiples, name):
        return tf.tile(input, multiples=multiples, name=name)

    @layer
    def squeeze_and_transpose(self, input, squeeze_dims, perm, name, conjugate=False):
        squeezed = tf.squeeze(input, squeeze_dims=squeeze_dims)
        out = tf.transpose(squeezed, perm=perm, name=name, conjugate=conjugate)
        return out

    @layer
    def image_resize(self, input, size, name, align_corners=True, method=tf.image.ResizeMethod.BILINEAR):
        if method=='nearest' or 'NEAREST_NEIGHBOR':
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        if method == 'bilinear' or 'BILINEAR':
            method = tf.image.ResizeMethod.BILINEAR
        return tf.image.resize_images(input, size, method=method, align_corners=align_corners)

    @layer
    def max_pool(self, input, pool_size, strides, name, padding=DEFAULT_PADDING):
        return tf.layers.max_pooling2d(input,
                                       pool_size=pool_size,
                                       strides=strides,
                                       padding=padding,
                                       name=name)

    @layer
    def avg_pool(self, input, pool_size, strides, name, padding=DEFAULT_PADDING):
        return tf.layers.average_pooling2d(input,
                                           pool_size=pool_size,
                                           strides=strides,
                                           padding=padding,
                                           name=name)

    @layer
    def l2_pool(self, input, pool_size, strides, name, padding=DEFAULT_PADDING):
        return tf.sqrt(tf.layers.average_pooling2d(
            tf.square(input),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            name=name))

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def multiply(self, inputs, name):
        return tf.multiply(inputs[0], inputs[1], name=name)

    @layer
    def multiply_channel_wise(self, inputs, name):
        '''  tile inputs[0] to match channel of inputs[1], and multiply element-wisely tiled [0] with [1]'''
        num_chan = tf.shape(inputs[1])[-1]
        return tf.multiply(tf.tile(inputs[0], [1,1,1,1,num_chan]), inputs[1], name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        return tf.layers.dense(input,
                               units=num_out,
                               activation=tf.nn.relu if relu else None,
                               trainable=self.trainable,
                               reuse=self.reuse,
                               name=name)

    @layer
    def sigmoid(self, input, name):
        return tf.sigmoid(input, name=name)

    @layer
    def softmax(self, input, name, dim=-1):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, dim=dim, name=name)

    @layer
    def nn_softmax(self, input, name, axis=-1):
        return tf.nn.softmax(input, axis=axis, name=name)

    @layer
    def batch_normalization(self, input, name, center=True, scale=True, relu=False):
        output = tf.layers.batch_normalization(input,
                                               center=center,
                                               scale=True,
                                               fused=True,
                                               trainable=self.trainable,
                                               reuse=self.reuse,
                                               name=name)
        if relu:
            output = tf.nn.relu(output)
        return output

    @layer
    def dropout(self, input, name):
        # return tf.layers.dropout(input,
        #                          rate=self.dropout_rate,
        #                          training=self.training,
        #                          seed=self.seed,
        #                          name=name)
        return slim.dropout(input,
                            keep_prob=self.dropout_rate,
                            is_training=self.training,
                            # seed=self.seed,
                            scope=name)

    @layer
    def l2norm(self, input, name, dim=-1):
        return tf.nn.l2_normalize(input, dim=dim, name=name)

    @layer
    def squeeze(self, input, axis=None, name=None):
        return tf.squeeze(input, axis=axis, name=name)

    @layer
    def expand_dims(self, input, axis, name=None):
        return tf.expand_dims(input, axis=axis, name=name)

