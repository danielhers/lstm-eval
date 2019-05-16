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
        n = len(domain)
        if distrib_name == 'uniform':
            return np.random.choice(a=domain, size=sample_size)

        elif distrib_name == 'u-shaped':
            alpha = 0.25
            beta = 0.25
            return np.random.choice(a=domain, size=sample_size, p=self.beta_bin_distrib(alpha, beta, n - 1))

        elif distrib_name == 'right-tailed':
            alpha = 1
            beta = 5
            return np.random.choice(a=domain, size=sample_size, p=self.beta_bin_distrib(alpha, beta, n - 1))

        elif distrib_name == 'left-tailed':
            alpha = 5
            beta = 1
            return np.random.choice(a=domain, size=sample_size, p=self.beta_bin_distrib(alpha, beta, n - 1))
        raise NotImplementedError(distrib_name)

    def generate_sample(self, sample_size=1, minv=1, maxv=50, distrib_type='uniform', distrib_display=False):
        input_arr = []
        output_arr = []

        # domain = [minv, ...., maxv]
        domain = list(range(minv, maxv + 1))

        nums = self.sample_from_a_distrib(domain, sample_size, distrib_type)

        for num in nums:
            i_seq = ''.join(elt for elt in self.vocabulary for _ in range(num))
            o_seq = ''
            for i in range(self.vocab_size):
                if i == 0:
                    o_seq += self.extra_letter * num  # a or b
                elif i == 1:
                    o_seq += self.vocabulary[i] * (num - 1)  # b
                else:
                    o_seq += self.vocabulary[i] * num  # other letters
            o_seq += 'T'  # termination symbol

            input_arr.append(i_seq)
            output_arr.append(o_seq)

        # Display the distribution of lengths of the samples
        if distrib_display:
            print('Distribution of the length of the samples: {}'.format(collections.Counter(nums)))

        return input_arr, output_arr, collections.Counter(nums)

    # Find letter index from all_letters
    def letter2index(self, letter):
        return self.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> tensor
    def letter2tensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letter2index(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def line2tensor_input(self, line):
        tensor = torch.zeros(len(line), 1, self.vocab_size)
        for li, letter in enumerate(line):
            if letter in self.all_letters:
                tensor[li][0][self.letter2index(letter)] = 1
            else:
                print('Error 1')
        return tensor

    def line2tensor_output(self, line):
        tensor = torch.zeros(len(line), self.n_letters)
        for li, letter in enumerate(line):
            if letter in self.all_letters:
                tensor[li][self.letter2index(letter)] = 1
            elif letter == self.extra_letter:  # a or b
                tensor[li][self.letter2index('a')] = 1
                tensor[li][self.letter2index('b')] = 1
            else:
                print('Error 2')
        return tensor
