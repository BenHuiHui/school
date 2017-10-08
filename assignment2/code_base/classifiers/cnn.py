from builtins import object
import numpy as np

from code_base.layers import *
from code_base.layer_utils import *
from code_base.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, dropout=0, seed=123, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.use_dropout = dropout > 0
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    #TODO: seed?

    C, H, W = input_dim
    W_con = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    b_con = np.zeros(num_filters)
    self.params['W1'] = W_con
    self.params['b1'] = b_con

    # TODO: how to get the size?

    filter_height = filter_size
    filter_width = filter_size
    stride_conv = 1  # stride
    P = (filter_size - 1) / 2  # padd
    Hc = (H + 2 * P - filter_height) / stride_conv + 1
    Wc = (W + 2 * P - filter_width) / stride_conv + 1

    width_pool = 2
    height_pool = 2
    stride_pool = 2
    Hp = (Hc - height_pool) / stride_pool + 1
    Wp = (Wc - width_pool) / stride_pool + 1

    W_hidden = np.random.normal(0, weight_scale, (num_filters * Hp * Wp, hidden_dim))
    b_hidden = np.zeros(hidden_dim)
    self.params['W2'] = W_hidden
    self.params['b2'] = b_hidden

    W_out = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    b_out = np.zeros(num_classes)
    self.params['W3'] = W_out
    self.params['b3'] = b_out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Set train/test mode for dropout param since it
    # behaves differently during training and testing.
    if self.use_dropout:
        self.dropout_param['mode'] = mode

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    convo_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

    N, F, H, W = convo_out.shape
    hidden_affine_out, hidden_affine_cahce = affine_forward(convo_out.reshape(N, -1), W2, b2)

    hidden_relu_out, hidden_relu_cache = relu_forward(hidden_affine_out)

    out_affine, out_affine_cache = affine_forward(hidden_relu_out, W3, b3)
    scores = out_affine

    # TODO: Is score the value after softmax?

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    data_loss, dx_softmax = softmax_loss(scores, y)

    reg_loss = 0.5 * self.reg * np.sum(W1 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W2 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W3 ** 2)

    loss = data_loss + reg_loss

    dx_out, dw_out, db_out = affine_backward(dx_softmax, out_affine_cache)
    dw_out += self.reg * W3

    grads['W3'] = dw_out
    grads['b3'] = db_out

    dx_hidden_relu = relu_backward(dx_out, hidden_relu_cache)
    dx_hidden_affine, dw_hidden_affine, db_hidden_affine = affine_backward(dx_hidden_relu, hidden_affine_cahce)
    dw_hidden_affine += self.reg * W2

    grads['W2'] = dw_hidden_affine
    grads['b1'] = db_hidden_affine

    dx_conv, dw_conv, db_conv = conv_relu_pool_backward(dx_hidden_affine.reshape(N, F, H, W), conv_cache)
    dw_conv += self.reg * W1

    grads['W1'] = dw_conv
    grads['b1'] = db_conv

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
