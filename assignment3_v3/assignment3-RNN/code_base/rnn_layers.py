from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################

    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    next_h = np.tanh(a)

    cache = (next_h, x, prev_h, Wx, Wh)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################

    next_h, x, prev_h, Wx, Wh = cache

    da = (1 - next_h**2) * dnext_h
    dx = np.dot(da, np.transpose(Wx))
    dprev_h = np.dot(da, np.transpose(Wh))
    dWx = np.dot(np.transpose(x), da)
    dWh = np.dot(np.transpose(prev_h), da)
    db = np.sum(da, axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def bidirectional_rnn_concatenate_forward(h, hr, mask):
    """
    (Optional) Forward pass for concatenating hidden vectors obtained from a RNN 
    trained on normal sentences and a RNN trained on reversed sentences at each 
    time step. We assume each hidden vector is of dimension H, timestep is T. 
    
    ***Important Note***
    The sentence length might be smaller than T, so there will be padding 0s at the 
    end of the sentences. For reversed RNN, the paddings are also at the end of the
    sentences, rather than at the begining of the sentences. For instance, given 5
    timesteps and a word vector representation of a sentence [s1, s2, s3, s4, 0]. 
    The vectors fed to two RNNs will be [s1, s2, s3, s4, 0] and [s4, s3, s2, s1, 0]
    , respectively, so will the order of the hidden states.

    Inputs:
    - h, hr: Input hidden states, of shape (N, T, H).
    - mask: Mask array which indices each sentence length, of shape (N, T), with
      0 paddings at the end.

    Returns a tuple of:
    - ho: Output vector for this timstep, shape (N, T, 2*H).
    - cache: Tuple of values needed for the backward pass.
    """
    ho, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for a single step of a bidirectional RNN. #
    ##############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return ho, cache


def bidirectional_rnn_concatenate_backward(dho, cache):
    """
    (Optional) Backward pass for a single timestep of the output layer of a 
    bidirectional RNN.

    ***Important Note***
    The sentence length might be smaller than T, so there will be padding 0s at the 
    end of the sentences. For reversed RNN, the paddings are also at the end of the
    sentences, rather than at the begining of the sentences. For instance, given 5
    timesteps and a word vector representation of a sentence [s1, s2, s3, s4, 0]. 
    The vectors fed to two RNNs will be [s1, s2, s3, s4, 0] and [s4, s3, s2, s1, 0]
    , respectively, so will the order of the hidden states. During backward pass,
    timesteps of the 0s will get 0 gradients.

    Inputs:
    - dout: Gradient of loss with respect to next hidden state.
    - cache: Cache object from the forward pass.

    Returns a tuple of:
    - dh, dhr: Gradients of input data, of shape (N, T, H).
    """
    dh, dhr = None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of the output layer of #
    #       of a bidirectional RNN.                                              #
    ##############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dh, dhr


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################

    N, T, D = x.shape
    _, H = h0.shape
    h = np.zeros((N, T, H))
    prev_h = h0
    cache = []

    for i in range(T):
        curr_x = x[:, i, :]  # .reshape(N, D)
        prev_h, c = rnn_step_forward(curr_x, prev_h, Wx, Wh, b)
        h[:, i, :] = prev_h
        cache.append(c)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################

    N, T, H = dh.shape
    D = cache[0][1].shape[1]
    dx, dh0, dWx, dWh, db = np.zeros((N, T, D)), np.zeros((N, H)), np.zeros((D, H)), np.zeros((H, H)), np.zeros(H)
    dh_next = np.zeros((N, H))

    for i in range(T-1, -1, -1):
        dx_curr, dprev_h_curr, dWx_curr, dWh_curr, db_curr = rnn_step_backward(dh[:, i, :], cache[i])
        dx_cul, dprev_h_cul, dWx_cul, dWh_cul, db_cul = rnn_step_backward(dh_next, cache[i])

        dh_next = dprev_h_cul + dprev_h_curr

        dx[:, i, :] = dx_cul + dx_curr
        dWx += dWx_cul + dWx_curr
        dWh += dWh_cul + dWh_curr
        db += db_cul + db_curr

    dh0 = dh_next

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def average_forward(hi, mask):
    """
    Forward pass to average the outputs at all timesteps. Assume we have T maximum
    timesteps.

    Inputs:
    - hi: Input data at all timesteps, of shape (N, T, H).
    - mask: Indicate the number of timesteps for each sample, i.e. sentence, of
            shape (N, T).

    Returns a tuple of:
    - ho: Averaged output vector, of shape (N, H).
    - cache: Tuple of values needed for the backward pass.
    """
    ho, cache = None, None
    N, T, H = hi.shape
    count = np.sum(mask, axis=1)
    ho = []
    for i in range(N):
        ho.append(np.dot(mask[i], hi[i]) / count[i])
    ho = np.array(ho)
    cache = {}
    cache['mask'] = mask
    cache['count'] = count
    return ho, cache


def average_backward(dho, cache):
    """
    Backward pass for the average layer.

    Inputs:
    - dho: Gradient of loss, of shape (N, H)
    - cache: Cache object from the forward pass.

    Returns a tuple of:
    - dhi: Gradients of input data, of shape (M, N, H).
    """
    N, H = dho.shape
    T = cache['mask'].shape[1]
    dhi = []
    for i in range(N):
        tmph = []
        for j in range(T):
            tmph.append(dho[i] * cache['mask'][i,j] / cache['count'][i])
        dhi.append(tmph)
    dhi = np.array(dhi)
    return dhi


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db
