from code_base.rnn_layers import rnn_step_forward, rnn_step_backward
from code_base.gradient_check import *
import numpy as np
from code_base.layer_utils import *
from code_base.classifiers.rnn import *


def verify_rnn_step_forward():
    from code_base.rnn_layers import rnn_step_forward
    from code_base.layer_utils import rel_error
    import numpy as np

    N, D, H = 3, 10, 4
    x = np.linspace(-0.4, 0.7, num=N * D).reshape(N, D)
    prev_h = np.linspace(-0.2, 0.5, num=N * H).reshape(N, H)
    Wx = np.linspace(-0.1, 0.9, num=D * H).reshape(D, H)
    Wh = np.linspace(-0.3, 0.7, num=H * H).reshape(H, H)
    b = np.linspace(-0.2, 0.4, num=H)
    next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
    expected_next_h = np.asarray([
        [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
        [0.66854692, 0.79562378, 0.87755553, 0.92795967],
        [0.97934501, 0.99144213, 0.99646691, 0.99854353]])
    print('next_h error: ', rel_error(expected_next_h, next_h))


def verify_rnn_step_backward():
    N, D, H = 4, 5, 6
    x = np.random.randn(N, D)
    h = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)
    out, cache = rnn_step_forward(x, h, Wx, Wh, b)
    dnext_h = np.random.randn(*out.shape)
    fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fh = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]
    dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
    dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
    db_num = eval_numerical_gradient_array(fb, b, dnext_h)
    dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)
    print('dx error: ', rel_error(dx_num, dx))
    print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
    print('dWx error: ', rel_error(dWx_num, dWx))
    print('dWh error: ', rel_error(dWh_num, dWh))
    print('db error: ', rel_error(db_num, db))


def verify_rnn_forward():
    from code_base.rnn_layers import rnn_forward
    import numpy as np

    N, T, D, H = 2, 3, 4, 5
    x = np.linspace(-0.1, 0.3, num=N * T * D).reshape(N, T, D)
    h0 = np.linspace(-0.3, 0.1, num=N * H).reshape(N, H)
    Wx = np.linspace(-0.2, 0.4, num=D * H).reshape(D, H)
    Wh = np.linspace(-0.4, 0.1, num=H * H).reshape(H, H)
    b = np.linspace(-0.7, 0.1, num=H)
    h, _ = rnn_forward(x, h0, Wx, Wh, b)
    expected_h = np.asarray([
        [[-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
         [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
         [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525], ],
        [[-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
         [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
         [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043]]])
    print('h error: ', rel_error(expected_h, h))


def verify_rnn_backward():
    from code_base.rnn_layers import rnn_forward, rnn_backward
    import numpy as np

    N, D, T, H = 2, 3, 10, 5
    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)
    out, cache = rnn_forward(x, h0, Wx, Wh, b)
    dout = np.random.randn(*out.shape)
    dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)
    fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
    fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
    fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
    fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
    fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]
    dx_num = eval_numerical_gradient_array(fx, x, dout)
    dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
    db_num = eval_numerical_gradient_array(fb, b, dout)
    print('dx error: ', rel_error(dx_num, dx))
    print('dh0 error: ', rel_error(dh0_num, dh0))
    print('dWx error: ', rel_error(dWx_num, dWx))
    print('dWh error: ', rel_error(dWh_num, dWh))
    print('db error: ', rel_error(db_num, db))


def verify_rnn_senti_forward():
    N, H, A, O = 2, 6, 5, 2
    word_to_idx = {'awesome': 0, 'reading': 1, 'pretty': 2, 'dog': 3, 'movie': 4,
                   'liked': 5, 'most': 6, 'admired': 7, 'bad': 8, 'fucking': 9}
    V = len(word_to_idx)
    T = 4
    model = SentimentAnalysisRNN(word_to_idx,
                                 hidden_dim=H,
                                 fc_dim=A,
                                 output_dim=O,
                                 cell_type='rnn',
                                 dtype=np.float64)
    # Set all model parameters to fixed values
    for k, v in model.params.items():
        model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)
    labels = np.array([1, 0], dtype=np.int32)
    wordvecs = np.zeros((N, T, V))
    wordvecs[0, 0, 0] = wordvecs[0, 1, 5] = wordvecs[0, 2, 2] = wordvecs[0, 3, 7] = 1
    wordvecs[1, 0, 4] = wordvecs[1, 1, 8] = wordvecs[1, 2, 5] = 1
    mask = np.ones((N, T))
    mask[1, 3] = 0
    loss, grads = model.loss(wordvecs, labels, mask)
    expected_loss = 2.99619226823
    # For brnn, the expected_loss should be 2.9577205234
    print('loss: ', loss)
    print('expected loss: ', expected_loss)
    print('difference: ', abs(loss - expected_loss))


def verify_rnn_senti_backward():
    N, T, H, A, O = 2, 4, 6, 5, 2
    word_to_idx = {'awesome': 0, 'reading': 1, 'pretty': 2, 'dog': 3, 'movie': 4,
                   'liked': 5, 'most': 6, 'admired': 7, 'bad': 8, 'fucking': 9}
    V = len(word_to_idx)
    labels = np.array([1, 0], dtype=np.int32)
    wordvecs = np.zeros((N, T, V))
    wordvecs[0, 0, 0] = wordvecs[0, 1, 5] = wordvecs[0, 2, 2] = wordvecs[0, 3, 7] = 1
    wordvecs[1, 0, 4] = wordvecs[1, 1, 8] = wordvecs[1, 2, 5] = 1
    mask = np.ones((N, T))
    mask[1, 3] = 0
    model = SentimentAnalysisRNN(word_to_idx,
                                 hidden_dim=H,
                                 fc_dim=A,
                                 output_dim=O,
                                 cell_type='rnn',
                                 dtype=np.float64,
                                 )
    loss, grads = model.loss(wordvecs, labels, mask)
    for param_name in sorted(grads):
        f = lambda _: model.loss(wordvecs, labels, mask)[0]
        param_grad_num = eval_numerical_gradient(f, model.params[param_name],
                                                 verbose=False, h=1e-6)
        e = rel_error(param_grad_num, grads[param_name])
        print('%s relative error: %e' % (param_name, e))


# verify_rnn_step_forward()
# verify_rnn_step_backward()
# verify_rnn_forward()
# verify_rnn_backward()
# verify_rnn_senti_forward()
verify_rnn_senti_backward()
