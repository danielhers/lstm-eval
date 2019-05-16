# Import relevant libraries and dependencies
import collections

import numpy as np
import torch
from scipy.special import gamma


# noinspection PyUnresolvedReferences
class SampleGenerator:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary  # _input vocabulary
        self.vocab_size = len(self.vocabulary)

        self.all_letters = vocabulary + 'T'  # _output vocabulary (T: termination symbol)
        self.letter2index = {elt: i for i, elt in enumerate(self.all_letters)}
        self.n_letters = len(self.all_letters)

        self.extra_letter = chr(ord(vocabulary[-1]) + 1)  # a or b (denoted a/b)

    def get_vocab(self):
        return self.vocabulary

    # Beta-Binomial density (pdf)
    @staticmethod
    def beta_binom_density(alpha, beta, k, n):
        return 1.0 * gamma(n + 1) * gamma(alpha + k) * gamma(n + beta - k) * gamma(alpha + beta) / (
                    gamma(k + 1) * gamma(n - k + 1) * gamma(alpha + beta + n) * gamma(alpha) * gamma(beta))

    # Beta-Binomial Distribution
    def beta_bin_distrib(self, alpha, beta, n):
        k = np.arange(n + 1, dtype=int)
        pdf = self.beta_binom_density(alpha, beta, k, n)

        # Normalize (to fix small precision errors)
        pdf *= (1. / pdf.sum())
        return pdf

    def sample_from_a_distrib(self, domain, sample_size, distrib_name):
        if distrib_name == 'uniform':
            return np.random.choice(a=domain, size=sample_size)
        elif distrib_name == 'u-shaped':
            alpha = beta = 0.25
        elif distrib_name == 'right-tailed':
            alpha = 1
            beta = 5
        elif distrib_name == 'left-tailed':
            alpha = 5
            beta = 1
        else:
            raise NotImplementedError(distrib_name)
        n = len(domain)
        return np.random.choice(a=domain, size=sample_size, p=self.beta_bin_distrib(alpha, beta, n - 1))

    def generate_sample(self, sample_size=1, minv=1, maxv=50, distrib_type='uniform', distrib_display=False):
        input_arr = []
        output_arr = []

        # domain = [minv, ...., maxv]
        domain = list(range(minv, maxv + 1))

        nums = self.sample_from_a_distrib(domain, sample_size, distrib_type)

        for num in nums:
            i_seq = ''.join(elt for elt in self.vocabulary for _ in range(num))
            o_seq = self.extra_letter * num  # a or b
            for i in range(1, self.vocab_size):
                o_seq += self.vocabulary[i] * ((num - 1) if i == 1 else num)  # b / other letters
            o_seq += 'T'  # termination symbol

            input_arr.append(i_seq)
            output_arr.append(o_seq)

        # Display the distribution of lengths of the samples
        if distrib_display:
            print('Distribution of the length of the samples: {}'.format(collections.Counter(nums)))

        return input_arr, output_arr, collections.Counter(nums)

    # Just for demonstration, turn a letter into a <1 x n_letters> tensor
    def letter2tensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letter2index[letter]] = 1
        return tensor

    # Turn lines into a <line_length x batch_size x n_letters>,
    # or an array of one-hot letter vectors
    def lines2tensor(self, lines, mode="input"):
        tensor = torch.zeros(max(map(len, lines)), len(lines), self.vocab_size if mode == "input" else self.n_letters)
        for i, line in enumerate(lines):
            for li, letter in enumerate(line):
                if mode == "output" and letter == self.extra_letter:
                    tensor[li][i][:-1] = 1
                else:
                    assert letter in self.all_letters, "Invalid letter " + letter
                    tensor[li][i][self.letter2index[letter]] = 1
        return tensor
