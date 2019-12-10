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
from keras.layers.recurrent import Recurrent,RNN
from keras.engine.base_layer import Layer
from keras import backend as K
from keras.engine import InputSpec
import warnings

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.legacy import interfaces




class MPO_LSTMCell_all(Layer):
    """
    # Arguments
        mpo_input_shape: a list of shapes, the product of which should be equal to the input dimension
        mpo_output_shape: a list of shapes of the same length as mpo_input_shape,
            the product of which should be equal to the output dimension
        mpo_ranks: a list of length len(mpo_input_shape)+1, the first and last rank should only be 1
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Tensor Train Recurrent Neural Networks for Video Classification](https://arxiv.org/abs/1707.01786)
    """

    def __init__(self,
                 w_input_shape, w_output_shape, w_ranks,
                 u_input_shape, u_output_shape, u_ranks,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 debug=False,
                 init_seed=11111986,
                 **kwargs):
        super(MPO_LSTMCell_all, self).__init__(**kwargs)

        self.units = np.prod(np.array(w_output_shape))
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self.state_spec = InputSpec(shape=(None, self.units))
        self.debug = debug
        self.init_seed = init_seed

        w_input_shape = np.array(w_input_shape)
        w_output_shape = np.array(w_output_shape)
        w_ranks = np.array(w_ranks)
        u_input_shape = np.array(u_input_shape)
        u_output_shape = np.array(u_output_shape)
        u_ranks = np.array(u_ranks)
        self.num_dim_w = w_input_shape.shape[0]
        self.w_input_shape_save = w_input_shape
        self.w_input_shape = w_input_shape
        self.w_output_shape_save = w_output_shape
        self.w_output_shape = w_output_shape
        self.w_ranks_save = w_ranks
        self.w_ranks = w_ranks

        self.num_dim_u = u_input_shape.shape[0]
        self.u_input_shape_save = u_input_shape
        self.u_input_shape = u_input_shape
        self.u_output_shape_save = u_output_shape
        self.u_output_shape = u_output_shape
        self.u_ranks_save = u_ranks
        self.u_ranks = u_ranks
        self.debug = debug

    def build(self, input_shape):

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[1]

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        input_dim = input_shape[1]
        self.input_dim = input_dim

        self.w_output_shape[0] *= 4
        self.u_output_shape[0] *= 4

        num_inputs = int(np.prod(input_shape[1::]))
        if np.prod(self.w_input_shape) != num_inputs:
            raise ValueError("The size of the input tensor (i.e. product "
                             "of the elements in w_input_shape) should "
                             "equal to the number of input neurons %d." %
                             (num_inputs))
        if self.w_input_shape.shape[0] != self.w_output_shape.shape[0]:
            raise ValueError("The number of input and output dimensions "
                             "should be the same.")
        if self.w_ranks.shape[0] != self.w_output_shape.shape[0] + 1:
            raise ValueError("The number of the w-ranks should be "
                             "1 + the number of the dimensions.")


        np.random.seed(self.init_seed)
        total_length_w = np.sum(self.w_input_shape * self.w_output_shape *
                               self.w_ranks[1:] * self.w_ranks[:-1])

        self.kernel_w = self.add_weight((total_length_w, ),
                                      initializer=self.kernel_initializer,
                                      name='kernel_w',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        total_length_u = np.sum(self.u_input_shape * self.u_output_shape *
                                self.u_ranks[1:] * self.u_ranks[:-1])

        self.kernel_u = self.add_weight((total_length_u,),
                                        initializer=self.kernel_initializer,
                                        name='kernel_u',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None


        # self.recurrent_kernel = self.add_weight(
        #     shape=(self.units, self.units*4),
        #     name='recurrent_kernel',
        #     initializer=self.recurrent_initializer,
        #     regularizer=self.recurrent_regularizer,
        #     constraint=self.recurrent_constraint)

        self.inds_w = np.zeros(self.num_dim_w).astype('int32')
        self.shapes_w = np.zeros((self.num_dim_w, 2)).astype('int32')
        self.cores_w = [None]*(self.num_dim_w)

        for k in range(self.num_dim_w -1, -1, -1):
            self.shapes_w[k] = (self.w_input_shape[k] * self.w_ranks[k + 1],
                              self.w_ranks[k] * self.w_output_shape[k])
            self.cores_w[k] = self.kernel_w[self.inds_w[k]:self.inds_w[k]+np.prod(self.shapes_w[k])]
            if 0 < k:
                self.inds_w[k-1] = self.inds_w[k] + np.prod(self.shapes_w[k])
        # print(self.num_dim_w)
        # print('self.shapes = ' + str(self.shapes_w))

        self.inds_u = np.zeros(self.num_dim_u).astype('int32')
        self.shapes_u = np.zeros((self.num_dim_u, 2)).astype('int32')
        self.cores_u = [None] * (self.num_dim_u)

        for k in range(self.num_dim_u - 1, -1, -1):
            self.shapes_u[k] = (self.u_input_shape[k] * self.u_ranks[k + 1],
                                self.u_ranks[k] * self.u_output_shape[k])
            self.cores_u[k] = self.kernel_u[self.inds_u[k]:self.inds_u[k] + np.prod(self.shapes_u[k])]
            if 0 < k:
                self.inds_u[k - 1] = self.inds_u[k] + np.prod(self.shapes_u[k])
        # print(self.num_dim_u)
        # print('self.shapes = ' + str(self.shapes_u))

        self.MPO_size = total_length_w + total_length_u
        self.full_size = (np.prod(self.w_input_shape) * np.prod(self.w_output_shape)) +\
                         (np.prod(self.u_input_shape) * np.prod(self.u_output_shape))
        self.compress_factor = 1. * self.MPO_size / self.full_size
        print('Compression factor = ' + str(self.MPO_size) + ' / ' \
              + str(self.full_size) + ' = ' + str(self.compress_factor))

        self.built = True

    def call(self, inputs, states,training=None):


        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0. < self.dropout < 1.:
            res = inputs * dp_mask[0]
        else:
            res = inputs

        for k in range(self.num_dim_w - 1, -1, -1):
            res = K.dot(K.reshape(res, (-1, self.shapes_w[k][0])),
                        K.reshape(self.cores_w[k], self.shapes_w[k])
                        )
            res = K.transpose(K.reshape(res, (-1, self.w_output_shape[k])))
        res = K.transpose(K.reshape(res, (-1, K.shape(inputs)[0])))

        z = res
        if 0. < self.recurrent_dropout < 1.:
            hid = h_tm1 * rec_dp_mask[0]
        else:
            hid = h_tm1
        for k in range(self.num_dim_u -1, -1, -1):
            hid = K.dot(K.reshape(hid, (-1, self.shapes_u[k][0])),
                        K.reshape(self.cores_u[k], self.shapes_u[k])
                        )
            hid = K.transpose(K.reshape(hid, (-1, self.u_output_shape[k])))
        hid = K.transpose(K.reshape(hid, (-1, K.shape(h_tm1)[0])))
        z += hid
        # z += K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0. < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]


    def get_config(self):
        config = {'w_input_shape':self.w_input_shape_save,
                  'w_output_shape':self.w_output_shape_save,
                  'w_ranks':self.w_ranks_save,
                  'u_input_shape': self.u_input_shape_save,
                  'u_output_shape': self.u_output_shape_save,
                  'u_ranks': self.u_ranks_save,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(MPO_LSTMCell_all, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MPO_LSTM_all(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Long short-term memory]
          (http://www.bioinf.jku.at/publications/older/2604.pdf)
        - [Learning to forget: Continual prediction with LSTM]
          (http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks]
          (http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in
           Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, w_input_shape,
                 w_output_shape,
                 w_ranks,
                 u_input_shape,
                 u_output_shape,
                 u_ranks,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        self.w_input_shape = w_input_shape
        self.w_output_shape = w_output_shape
        self.w_ranks = w_ranks
        self.u_input_shape = u_input_shape
        self.u_output_shape = u_output_shape
        self.u_ranks = u_ranks
        cell = MPO_LSTMCell_all(w_input_shape,
                        w_output_shape,
                        w_ranks,
                        u_input_shape,
                        u_output_shape,
                        u_ranks,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout)

        super(MPO_LSTM_all, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(MPO_LSTM_all, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout


    def get_config(self):
        config = {'w_input_shape': self.w_input_shape,
                  'w_output_shape': self.w_output_shape,
                  'w_ranks': self.w_ranks,
                  'u_input_shape': self.u_input_shape,
                  'u_output_shape': self.u_output_shape,
                  'u_ranks': self.u_ranks,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(MPO_LSTM_all, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)



def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)


def _standardize_args(inputs, initial_state, constants, num_constants):
    """Standardize `__call__` to a single list of tensor inputs.

    When running a model loaded from file, the input tensors
    `initial_state` and `constants` can be passed to `RNN.__call__` as part
    of `inputs` instead of by the dedicated keyword arguments. This method
    makes sure the arguments are separated and that `initial_state` and
    `constants` are lists of tensors (or None).

    # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None

    # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
    """
    if isinstance(inputs, list):
        assert initial_state is None and constants is None
        if num_constants is not None:
            constants = inputs[-num_constants:]
            inputs = inputs[:-num_constants]
        if len(inputs) > 1:
            initial_state = inputs[1:]
        inputs = inputs[0]

    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]

    initial_state = to_list_or_none(initial_state)
    constants = to_list_or_none(constants)

    return inputs, initial_state, constants
