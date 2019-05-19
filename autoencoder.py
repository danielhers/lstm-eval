from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn

from model import MyLSTM


def generate_sample(vocab, num_samples, max_len):
    for sample_len in np.random.randint(1, max_len + 1, num_samples):
        yield np.random.choice(vocab, sample_len)


def one_hot(a, vocab):
    return torch.zeros(a.shape + (len(vocab),)).scatter(-1, np.vectorize(vocab.find)(a), 1)


# noinspection PyUnresolvedReferences
def main(args):
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    vocab = "abc"
    for trial_num in range(args.trials):
        samples = np.array(generate_samples(vocab, args.sample, args.maxlen))
        inputs = one_hot(samples, vocab).to(device)
        targets = one_hot(samples[:, :1]).to(device)
        lstm = MyLSTM(args.dim, len(vocab), args.layers).to(device)
        criterion = nn.MSELoss()
        optim = torch.optim.RMSprop(lstm.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            lstm.zero_grad()
            h0, c0 = lstm.init_hidden(args.sample)
            outputs, _ = lstm(inputs, (h0.to(device), c0.to(device)))
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
        path = 'models/autoencoder_{}.pth'.format(trial_num)
        torch.save(lstm, path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use if CUDA is available.')
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument('--dim', type=int, default=[3], nargs='*',
                        help='A list of hidden units for our LSTM for a given language (e.g. 4 10 36).')
    parser.add_argument('--layers', type=int, default=1, help='The number of layers in the LSTM network.')
    parser.add_argument('--epochs', type=int, default=150, help='The total number of epochs.')
    parser.add_argument('--sample', type=int, default=1000, help='The total number of training samples.')
    parser.add_argument('--maxlen', type=int, default=1000, help='Maximum sample length.')
    main(parser.parse_args())
