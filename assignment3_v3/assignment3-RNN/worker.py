from code_base.rnn_layers import rnn_step_forward
import numpy as np
from code_base.sentiment_analysis_solver import *
from code_base.classifiers.rnn import *
# If you do brnn, please import from code_base.classifiers.brnn instead
from code_base.data_utils import *
import matplotlib.pyplot as plt


def q1():
    x_shape = (3, 874)
    Wx_shape = (874, 128)
    h_shape = (3, 128)
    Wh_shape = (128, 128)
    b_shape = (128,)
    x = np.loadtxt('./input_files/x.csv', delimiter=',')
    x = x.reshape(x_shape)
    Wx = np.loadtxt('./input_files/Wx.csv', delimiter=',')
    Wx = Wx.reshape(Wx_shape)
    prev_h = np.loadtxt('./input_files/prev_h.csv', delimiter=',')
    prev_h = prev_h.reshape(h_shape)
    Wh = np.loadtxt('./input_files/Wh.csv', delimiter=',')
    Wh = Wh.reshape(Wh_shape)
    b = np.loadtxt('./input_files/b.csv', delimiter=',')
    out, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
    np.savetxt('./output_files/rnn_step_forward_out.csv', out.ravel(), delimiter=',')


def q2():
    from code_base.rnn_layers import rnn_step_forward, rnn_step_backward
    import numpy as np

    x_shape = (3, 874)
    Wx_shape = (874, 128)
    h_shape = (3, 128)
    Wh_shape = (128, 128)
    b_shape = (128,)
    x = np.loadtxt('./input_files/x.csv', delimiter=',')
    x = x.reshape(x_shape)
    Wx = np.loadtxt('./input_files/Wx.csv', delimiter=',')
    Wx = Wx.reshape(Wx_shape)
    prev_h = np.loadtxt('./input_files/prev_h.csv', delimiter=',')
    prev_h = prev_h.reshape(h_shape)
    Wh = np.loadtxt('./input_files/Wh.csv', delimiter=',')
    Wh = Wh.reshape(Wh_shape)
    b = np.loadtxt('./input_files/b.csv', delimiter=',')
    out, cache = rnn_step_forward(x, prev_h, Wx, Wh, b)
    dhout = np.loadtxt('./input_files/dho.csv', delimiter=',')
    dx, dh, dWx, dWh, db = rnn_step_backward(dhout, cache)
    np.savetxt('./output_files/rnn_step_backward_out_dx.csv', dx.ravel(), delimiter=',')
    np.savetxt('./output_files/rnn_step_backward_out_dh.csv', dh.ravel(), delimiter=',')
    np.savetxt('./output_files/rnn_step_backward_out_dwx.csv', dWx.ravel(), delimiter=',')
    np.savetxt('./output_files/rnn_step_backward_out_dwh.csv', dWh.ravel(), delimiter=',')
    np.savetxt('./output_files/rnn_step_backward_out_db.csv', db.ravel(), delimiter=',')


def q3():
    from code_base.rnn_layers import rnn_forward
    import numpy as np

    x_all_shape = (3, 5, 874)
    Wx_shape = (874, 128)
    h_shape = (3, 128)
    Wh_shape = (128, 128)
    b_shape = (128,)
    x_all = np.loadtxt('./input_files/x_all.csv', delimiter=',')
    x_all = x_all.reshape(x_all_shape)
    Wx = np.loadtxt('./input_files/Wx.csv', delimiter=',')
    Wx = Wx.reshape(Wx_shape)
    h0 = np.loadtxt('./input_files/prev_h.csv', delimiter=',')
    h0 = h0.reshape(h_shape)
    Wh = np.loadtxt('./input_files/Wh.csv', delimiter=',')
    Wh = Wh.reshape(Wh_shape)
    b = np.loadtxt('./input_files/b.csv', delimiter=',')
    out, _ = rnn_forward(x_all, h0, Wx, Wh, b)
    np.savetxt('./output_files/rnn_forward_out.csv', out.ravel(), delimiter=',')


def q4():
    from code_base.rnn_layers import rnn_forward, rnn_backward
    import numpy as np

    x_all_shape = (3, 5, 874)
    Wx_shape = (874, 128)
    h_shape = (3, 128)
    Wh_shape = (128, 128)
    b_shape = (128,)
    dh_all_shape = (3, 5, 128)
    x_all = np.loadtxt('./input_files/x_all.csv', delimiter=',')
    x_all = x_all.reshape(x_all_shape)
    Wx = np.loadtxt('./input_files/Wx.csv', delimiter=',')
    Wx = Wx.reshape(Wx_shape)
    h0 = np.loadtxt('./input_files/prev_h.csv', delimiter=',')
    h0 = h0.reshape(h_shape)
    Wh = np.loadtxt('./input_files/Wh.csv', delimiter=',')
    Wh = Wh.reshape(Wh_shape)
    b = np.loadtxt('./input_files/b.csv', delimiter=',')
    out, cache = rnn_forward(x_all, h0, Wx, Wh, b)
    dhout = np.loadtxt('./input_files/dho_all.csv', delimiter=',')
    dhout = dhout.reshape(dh_all_shape)
    dx_all, dh0, dWx, dWh, db = rnn_backward(dhout, cache)
    np.savetxt('./output_files/rnn_backward_out_dx.csv', dx_all.ravel(), delimiter=',')
    np.savetxt('./output_files/rnn_backward_out_dh0.csv', dh0.ravel(), delimiter=',')
    np.savetxt('./output_files/rnn_backward_out_dwx.csv', dWx.ravel(), delimiter=',')
    np.savetxt('./output_files/rnn_backward_out_dwh.csv', dWh.ravel(), delimiter=',')
    np.savetxt('./output_files/rnn_backward_out_db.csv', db.ravel(), delimiter=',')


def q5():
    download_corpus()
    small_data = load_data('code_base/datasets/train.csv', sample=True)
    small_rnn_model = SentimentAnalysisRNN(
        cell_type='rnn',
        word_to_idx=load_dictionary('code_base/datasets/dictionary.csv')
    )
    small_rnn_solver = SentimentAnalysisSolver(small_rnn_model,
                                               small_data,
                                               update_rule='sgd',
                                               num_epochs=100,
                                               batch_size=100,
                                               optim_config={
                                                   'learning_rate': 5e-3,
                                               },
                                               lr_decay=1.0,
                                               verbose=True,
                                               print_every=10,
                                               )
    small_rnn_solver.train()

    # we will use the same batch of training data for inference
    # this is just to let you know the procedure of inference
    preds = small_rnn_solver.test(split='train')
    np.savetxt('./output_files/rnn_prediction_prob.csv', preds.ravel(), delimiter=',')
    # If you do brnn, please save result to ./output_files/brnn_prediction_prob.csv

    # Plot the training losses
    plt.plot(small_rnn_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()


# q1()
# q2()
# q3()
# q4()
q5()
