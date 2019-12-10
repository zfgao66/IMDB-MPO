__author__ = "Ze-Feng Gao"
__copyright__ = "Siemens AG, 2019"
__licencse__ = "MIT"
__version__ = "0.1"

"""
MIT License

Copyright (c) 2019 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np

import keras.activations
from keras import backend as K
from keras.engine.topology import Layer

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints



class MPO_Layer(Layer):
    """
    # Arguments:
        mpo_input_shape: a list of shapes, the product of which should be equal to the input dimension
        mpo_output_shape: a list of shapes of the same length as mpo_input_shape,
            the product of which should be equal to the output dimension
        mpo_ranks: a list of length len(mpo_input_shape)+1, the first and last rank should only be 1
        the rest of the arguments: please refer to dense layer in keras. 
    """

    def __init__(self, mpo_input_shape, mpo_output_shape, mpo_ranks,
                 use_bias=True,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 debug=False,
                 init_seed=11111986,
                 **kwargs):

        mpo_input_shape = np.array(mpo_input_shape)
        mpo_output_shape = np.array(mpo_output_shape)
        mpo_ranks = np.array(mpo_ranks)

        self.mpo_input_shape = mpo_input_shape
        self.mpo_output_shape = mpo_output_shape
        self.mpo_ranks = mpo_ranks
        self.num_dim = mpo_input_shape.shape[0]  # length of the train
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.debug = debug
        self.init_seed = init_seed

        super(MPO_Layer, self).__init__(**kwargs)

    def build(self, input_shape):

        num_inputs = int(np.prod(input_shape[1::]))

        # Check the dimensionality
        if np.prod(self.mpo_input_shape) != num_inputs:
            raise ValueError("The size of the input tensor (i.e. product "
                             "of the elements in mpo_input_shape) should "
                             "equal to the number of input neurons %d." % num_inputs)
        if self.mpo_input_shape.shape[0] != self.mpo_output_shape.shape[0]:
            raise ValueError("The number of input and output dimensions "
                             "should be the same.")
        if self.mpo_ranks.shape[0] != self.mpo_output_shape.shape[0] + 1:
            raise ValueError("The number of the MPO-ranks should be "
                             "1 + the number of the dimensions.")
        if self.debug:
            print('mpo_input_shape = ' + str(self.mpo_input_shape))
            print('mpo_output_shape = ' + str(self.mpo_output_shape))
            print('mpo_ranks = ' + str(self.mpo_ranks))

        # Initialize the weights
        if self.init_seed is None:
            self.init_seed = 11111986
        np.random.seed(self.init_seed)


        total_length = np.sum(self.mpo_input_shape * self.mpo_output_shape *
                                  self.mpo_ranks[1:] * self.mpo_ranks[:-1])

        self.kernel = self.add_weight((total_length, ),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight((np.prod(self.mpo_output_shape), ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        # Pre-calculate the indices, shapes and cores
        self.inds = np.zeros(self.num_dim).astype('int32')
        self.shapes = np.zeros((self.num_dim, 2)).astype('int32')
        self.cores = [None] * self.num_dim

        for k in range(self.num_dim - 1, -1, -1):
            # This is the shape of (m_k * r_{k+1}) * (r_k * n_k)
            self.shapes[k] = (self.mpo_input_shape[k] * self.mpo_ranks[k + 1],
                              self.mpo_ranks[k] * self.mpo_output_shape[k])
            # Note that self.cores store only the pointers to the parameter vector
            self.cores[k] = self.kernel[self.inds[k]:self.inds[k] + np.prod(self.shapes[k])]
            if 0 < k:  # < self.num_dim-1:
                self.inds[k - 1] = self.inds[k] + np.prod(self.shapes[k])
        if self.debug:
            print('self.shapes = ' + str(self.shapes))

        # Calculate and print the compression factor
        self.MPO_size = total_length
        self.full_size = (np.prod(self.mpo_input_shape) * np.prod(self.mpo_output_shape))
        self.compress_factor = 1. * self.MPO_size / self.full_size
        print('Compression factor = ' + str(self.MPO_size) + ' / ' \
              + str(self.full_size) + ' = ' + str(self.compress_factor))

    def call(self, x, mask=None):
        res = x
        for k in range(self.num_dim - 1, -1, -1):

            # New one, in order to avoid calculating the indices in every iteration
            res = K.dot(K.reshape(res, (-1, self.shapes[k][0])),  # of shape (-1, m_k*r_{k+1})
                        K.reshape(self.cores[k], self.shapes[k])  # of shape (m_k*r_{k+1}, r_k*n_k)
                        )
            res = K.transpose(
                K.reshape(res, (-1, self.mpo_output_shape[k]))
            )

        # res is of size o_1 x ... x o_d x batch_size # by Alexander
        res = K.transpose(K.reshape(res, (-1, K.shape(x)[0])))

        if self.use_bias:
            res = K.bias_add(res, self.bias)
        if self.activation is not None:
            res =self.activation(res)

        return res

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(self.mpo_output_shape))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(self.mpo_output_shape))

    def _generate_orthogonal_mpo_cores(self):
        cores_arr_len = np.sum(self.mpo_input_shape * self.mpo_output_shape *
                               self.mpo_ranks[1:] * self.mpo_ranks[:-1])
        cores_arr = np.zeros(cores_arr_len)
        rv = 1

        d = self.mpo_input_shape.shape[0]
        rng = np.random
        shapes = [None] * d
        tall_shapes = [None] * d
        cores = [None] * d
        counter = 0

        for k in range(self.mpo_input_shape.shape[0]):
            # Original implementation
            # shape = [ranks[k], input_shape[k], output_shape[k], ranks[k+1]]
            shapes[k] = [self.mpo_ranks[k], self.mpo_input_shape[k], self.mpo_output_shape[k], self.mpo_ranks[k + 1]]

            # Original implementation
            # tall_shape = (np.prod(shape[:3]), shape[3])
            tall_shapes[k] = (np.prod(shapes[k][:3]), shapes[k][3])

            # Original implementation
            # curr_core = np.dot(rv, np.random.randn(shape[0], np.prod(shape[1:])) )
            cores[k] = np.dot(rv, rng.randn(shapes[k][0], np.prod(shapes[k][1:])))

            # Original implementation
            # curr_core = curr_core.reshape(tall_shape)
            cores[k] = cores[k].reshape(tall_shapes[k])

            if k < self.mpo_input_shape.shape[0] - 1:
                # Original implementation
                # curr_core, rv = np.linalg.qr(curr_core)
                cores[k], rv = np.linalg.qr(cores[k])
            # Original implementation
            # cores_arr[cores_arr_idx:cores_arr_idx+curr_core.size] = curr_core.flatten()
            # cores_arr_idx += curr_core.size
            cores_arr[counter:(counter + cores[k].size)] = cores[k].flatten()
            counter += cores[k].size

        glarot_style = (np.prod(self.mpo_input_shape) * np.prod(self.mpo_ranks)) ** (1.0 / self.mpo_input_shape.shape[0])
        return (0.1 / glarot_style) * cores_arr

    def get_full_W(self):
        res=np.identity(np.prod(self.mpo_input_shape))
        for k in range(self.num_dim - 1, -1, -1):
            res = np.dot(np.reshape(res, (-1, self.shapes[k][0])),  # of shape (-1, m_k*r_{k+1})
                        np.reshape(self.cores[k], self.shapes[k])  # of shape (m_k*r_{k+1}, r_k*n_k)
                        )
            res = np.transpose(
                np.reshape(res, (-1, self.mpo_output_shape[k]))
            )
        res = np.transpose(np.reshape(res, (-1, np.shape(res)[0])))

        if self.use_bias:
            res = K.bias_add(res, self.bias)
        if self.activation is not None:
            res =self.activation(res)

        return res

    def get_config(self):
        config = {'mpo_input_shape':self.mpo_input_shape,
                  'mpo_output_shape':self.mpo_output_shape,
                  'mpo_ranks':self.mpo_ranks,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(MPO_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

