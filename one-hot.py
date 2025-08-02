import numpy as np


def one_hot_encode(sequence):
    en_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
    en_seq = [en_dict[ch] for ch in sequence]
    np_seq = np.array(en_seq, dtype=int)
    seq_emb = np.zeros((len(np_seq), 5))
    seq_emb[np.arange(len(np_seq)), np_seq] = 1
    return seq_emb.astype(np.float32)


if __name__ == '__main__':
    seq = 'ACGT'
    print(one_hot_encode(seq))

