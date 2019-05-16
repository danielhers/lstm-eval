# Import relevant libraries and dependencies
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model import MyLSTM
from sample_generator import SampleGenerator

MAX_INT = sys.maxsize

# Default value: 5
first_k_errors = 5

# Epsilon value -- output threshold (during test time)
epsilon = 0.5


def get_args():
    parser = argparse.ArgumentParser(description='Let us train an LSTM model.')

    # Experiment type
    parser.add_argument('--exp_type', required=True, type=str, default='single',
                        choices=['single', 'distribution', 'window', 'hidden_units'], help='The experiment type.')

    # Required params
    parser.add_argument('--language', type=str, default='abc', choices=['ab', 'abc', 'abcd'],
                        help='The language in consideration.')
    parser.add_argument('--distribution', type=str, default=['uniform'], nargs='*',
                        choices=['uniform', 'u-shaped', 'left-tailed', 'right-tailed'],
                        help='A list of distribution regimes for our training set (e.g. \'uniform\' \'u_shaped\' \'left'
                             '_tailed\' \'right_tailed\').')
    parser.add_argument('--window', type=int, default=[1, 50], nargs='*',
                        help='A list of length windows for our training set (e.g. 1 30 1 50 50 100 for 1-30, 1-50, '
                             '50-100).')
    parser.add_argument('--lstm_hunits', type=int, default=[3], nargs='*',
                        help='A list of hidden units for our LSTM for a given language (e.g. 4 10 36).')

    # Optional params
    parser.add_argument('--lstm_hlayers', type=int, default=1, help='The number of layers in the LSTM network.')
    parser.add_argument('--sample_size', type=int, default=1000, help='The total number of training samples.')
    parser.add_argument('--n_epochs', type=int, default=150, help='The total number of epochs.')
    parser.add_argument('--n_trials', type=int, default=10, help='The total number of trials.')
    parser.add_argument('--disp_err_n', type=int, default=5, help='The first k error values.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use if CUDA is available.')

    params, _ = parser.parse_known_args()

    # Print the entire list of parameters
    print(params, flush=True)

    return params


def plot_graphs(lang, labels, accuracy_vals, window, filename):
    accuracy_vals = np.array(accuracy_vals)

    domain = list(range(1, accuracy_vals.shape[2] + 1))

    # For plotting purposes...
    # Uncomment the following line if you would like to bound the plot window by the maximum e_k value (Ref 1)
    # max_y = np.max(accuracy_vals) + 10

    e_nums = [1, first_k_errors]

    for err_n in e_nums:
        plt.figure()
        for i in range(len(labels)):
            acc = np.array(accuracy_vals[i])
            acc_avg = np.average(acc, axis=0).T
            plt.plot(domain, acc_avg[err_n - 1], '.-', label=labels[i])

        plt.legend(loc='upper left')

        if window is not None:
            # Lower training threshold
            plt.plot(domain, np.ones(len(domain)) * window[0], 'c-', label='Threshold$_1$')
            # Upper training threshold
            plt.plot(domain, np.ones(len(domain)) * window[1], 'c-', label='Threshold$_2$')

        lang_str = '^n'.join(lang + ' ')[:-1]

        plt.title('Generalization Graph for ${}$'.format(lang_str))
        plt.xlabel('Epoch Number')
        plt.ylabel('$e_{}$ Value'.format(str(err_n)))
        # For plotting purposes (Ref 1)
        # plt.ylim ([0, max_y])
        fig_filename = 'figures/{}_error_{}'.format(filename, str(err_n))
        plt.savefig(fig_filename, dpi=256)
        print("Saved {}.png".format(fig_filename), flush=True)
    return


def single_investigation(lang, distrib, h_layers, h_units, window, sample_size, n_epochs, exp_num, device):
    acc_per_d = []
    loss_per_d = []
    loss_vals = None

    generator = SampleGenerator(lang)
    # If you would like to fix your training set during the entire course of investigation, 
    # you should uncomment the following line (and comment the same line in the subsequent "for" loop);
    # otherwise, each training set will come from the same distribution and same window but be different.
    inputs, outputs, s_dst = generator.generate_sample(sample_size, window[0], window[1], distrib, False)
    for _ in range(exp_num):
        # inputs, outputs, s_dst = generator.generate_sample (sample_size, window[0], window[1], distrib, False)
        e_vals, loss_vals = train(generator, distrib, h_layers, h_units, inputs, outputs, n_epochs, 1,
                                  device)  # each experiment is unique
        acc_per_d.append(e_vals)
        loss_per_d.append(loss_vals)

    filename = '{}_{}_{}_{}_{}_{}_{}'.format(lang, 'single', distrib, h_layers, h_units, window[0], window[1])

    # Uncomment the following line if you would like to save the e_i and loss values.
    # np.savez('./results/result_{}.npz'.format(filename), errors = np.array(e_vals), losses = np.array (loss_vals))

    trials_label = ['Experiment {}'.format(elt) for elt in range(1, exp_num + 1)]
    plot_graphs(lang, trials_label, acc_per_d, window, filename)

    return acc_per_d, loss_vals


def hidden_units_investigation(lang, distrib, h_layers, h_units, window, sample_size, n_epochs, exp_num, device):
    acc_per_d = []
    loss_per_d = []
    loss_vals = None

    generator = SampleGenerator(lang)
    # If you would like to fix your training set during the entire course of investigation, 
    # you should uncomment the following line (and comment the same line in the subsequent "for" loop);
    # otherwise, each training set will come from the same distribution and same window but be different.
    inputs, outputs, s_dst = generator.generate_sample(sample_size, window[0], window[1], distrib, False)
    for hidden_dim in h_units:
        print(hidden_dim, flush=True)
        # inputs, outputs, s_dst = generator.generate_sample (sample_size, window[0], window[1], distrib, False)
        e_vals, loss_vals = train(generator, distrib, h_layers, hidden_dim, inputs, outputs, n_epochs, exp_num, device)
        acc_per_d.append(e_vals)
        loss_per_d.append(loss_vals)

    filename = '{}_{}_{}_{}_{}_{}'.format(lang, 'hidden', distrib, h_layers, window[0], window[1])
    hunits_label = ['{} Hidden Units'.format(val) for val in h_units]
    plot_graphs(lang, hunits_label, acc_per_d, window, filename)

    return acc_per_d, loss_vals


def window_investigation(lang, distrib, h_layers, h_units, windows, sample_size, n_epochs, exp_num, device):
    acc_per_d = []
    loss_per_d = []
    loss_vals = None

    generator = SampleGenerator(lang)
    for window in windows:
        print(window, flush=True)
        filename = 'results/{}_{}_{}_{}_{}.npz'.format(lang, window, distrib, h_layers, h_units)
        if os.path.exists(filename):
            d = np.load(filename)
            e_vals = d["errors"]
            loss_vals = d["losses"]
        else:
            inputs, outputs, s_dst = generator.generate_sample(sample_size, window[0], window[1], distrib, False)
            e_vals, loss_vals = train(generator, distrib, h_layers, h_units, inputs, outputs, n_epochs, exp_num, device)
            np.savez(filename, errors=np.array(e_vals), losses=np.array(loss_vals))
        acc_per_d.append(e_vals)
        loss_per_d.append(loss_vals)

    filename = '{}_{}_{}_{}_{}'.format(lang, 'window', distrib, h_layers, h_units)
    window_label = ['Window [{}, {}]'.format(elt[0], elt[1]) for elt in windows]
    plot_graphs(lang, window_label, acc_per_d, None, filename)

    return acc_per_d, loss_vals


def distribution_investigation(lang, distribution, h_layers, h_units, window, sample_size, n_epochs, exp_num, device):
    acc_per_d = []
    loss_per_d = []
    loss_vals = None

    generator = SampleGenerator(lang)
    for distrib in distribution:
        print(distrib, flush=True)
        inputs, outputs, s_dst = generator.generate_sample(sample_size, window[0], window[1], distrib, False)
        e_vals, loss_vals = train(generator, distrib, h_layers, h_units, inputs, outputs, n_epochs, exp_num, device)
        acc_per_d.append(e_vals)
        loss_per_d.append(loss_vals)

    filename = '{}_{}_{}_{}_{}_{}'.format(lang, 'distrib', h_layers, h_units, window[0], window[1])
    distrib_label = [elt.capitalize() for elt in distribution]
    plot_graphs(lang, distrib_label, acc_per_d, window, filename)

    return acc_per_d, loss_vals


# noinspection PyUnresolvedReferences
def test(generator, lstm, device):
    first_errors = []

    with torch.no_grad():
        for num in range(1, MAX_INT):
            inputs, outputs, _ = generator.generate_sample(1, num, num)
            inp, out = inputs[0], outputs[0]
            input_size = len(inp)

            h0, c0 = lstm.init_hidden()
            output, hidden = lstm(generator.lines2tensor([inp]).to(device), (h0.to(device), c0.to(device)))
            output = output.cpu()
            predictions = np.int_(output.numpy() >= epsilon)
            actual = np.int_(generator.lines2tensor([out], mode="output").to(device).cpu().numpy())

            if not np.all(np.equal(predictions, actual)):
                first_errors.append(num)
                if len(first_errors) == first_k_errors:
                    return first_errors


# noinspection PyUnresolvedReferences
def train(generator, distrib, h_layers, h_units, inputs, outputs, n_epochs, exp_num, device):
    lang = generator.get_vocab()
    vocab_size = len(lang)
    training_size = len(inputs)

    loss_arr_per_iter = []
    first_errors_per_iter = []
    os.makedirs("models", exist_ok=True)

    for exp in range(exp_num):
        print('Experiment Number: {}'.format(exp + 1), flush=True)

        # Create the model
        lstm = MyLSTM(h_units, vocab_size, h_layers).to(device)
        # In our experiments, a value between 0.01 and 0.001 worked well
        learning_rate = .01  # learning rate 

        criterion = nn.MSELoss()  # MSE Loss
        optim = torch.optim.RMSprop(lstm.parameters(), lr=learning_rate)  # RMSProp optimizer

        loss_arr = []
        first_errors = []

        for it in range(1, n_epochs + 1):
            # total loss per epoch
            lstm.zero_grad()
            h0, c0 = lstm.init_hidden(training_size)
            output, _ = lstm(generator.lines2tensor(inputs).to(device), (h0.to(device), c0.to(device)))
            target = generator.lines2tensor(outputs, mode="output").to(device)
            # loss for a single sample
            loss = criterion(output, target)
            loss.backward()
            optim.step()
            loss_arr.append(loss.item())  # add loss val
            first_errors.append(test(generator, lstm, device))  # add e_i vals

        loss_arr_per_iter.append(loss_arr)
        first_errors_per_iter.append(first_errors)

        # print ('Loss array: ', loss_arr)
        # print ('Max Gen: ', first_errors)

        # We can save the models as we train.
        rnn_path = 'models/lstm_lang{}_distrib_{}_expn_{}.pth'.format(lang, distrib, str(exp))
        torch.save(lstm, rnn_path)

    return first_errors_per_iter, loss_arr_per_iter


# noinspection PyUnresolvedReferences
def main(args):
    global first_k_errors
    investigation = args.exp_type
    lang = args.language
    distrib = args.distribution
    window = []
    for i in range(int(len(args.window) / 2)):
        window.append([args.window[2 * i], args.window[2 * i + 1]])

    n_units = args.lstm_hunits
    n_layers = args.lstm_hlayers
    s_size = args.sample_size
    n_epochs = args.n_epochs
    n_trials = args.n_trials
    first_k_errors = args.disp_err_n

    # For GPU usage
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    if investigation == 'distribution':
        distribution_investigation(lang, distrib, n_layers, n_units[0], window[0], s_size, n_epochs, n_trials, device)
    elif investigation == 'window':
        window_investigation(lang, distrib[0], n_layers, n_units[0], window, s_size, n_epochs, n_trials, device)
    elif investigation == 'hidden_units':
        hidden_units_investigation(lang, distrib[0], n_layers, n_units, window[0], s_size, n_epochs, n_trials, device)
    elif investigation == 'single':
        single_investigation(lang, distrib[0], n_layers, n_units[0], window[0], s_size, n_epochs, n_trials, device)
    else:
        print('Sorry, we couldn\'t process your input; could you please try again?')

    print('\nGoodbye!..\n')
    return


if __name__ == "__main__":
    main(get_args())
