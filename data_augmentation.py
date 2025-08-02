import random
import math

rule = {
    'A': 'T',
    'C': 'G',
    'T': 'A',
    'G': 'C',
    'N': 'N'
}


def slicing(seq, input_feature, output_feature,seq_len):
    '''
    Perform window slicing on the sequence
    :param seq: orial sequenue
    :param input_feature: input data feature
    :param output_feature: output data feature
    :return: seq[], input_feature[], output_feature[]
    '''
    input_feature = input_feature.split(' ')
    #input_feature_2 = input_feature_2.split(' ')
    output_feature = output_feature.split(' ')

    count_N = seq.count('N')
    #print(seq)
    end = len(seq) - count_N
    random_integer_1 = random.randint(0, int(end / 2))
    random_integer_2 = random.randint(0, int(end / 2))
    while math.fabs(random_integer_1 - random_integer_2) < seq_len/10:
        print(random_integer_2,random_integer_1,int(end / 2),seq_len/10)
        random_integer_2 = random.randint(0, int(end / 2))

    seq_one = seq[random_integer_1:random_integer_1 + int(end / 2)]
    seq_two = seq[random_integer_2:random_integer_2 + int(end / 2)]

    length = len(seq_one)

    input_feature_one = input_feature[random_integer_1:
                                      random_integer_1 + int(end / 2)]
    input_feature_two = input_feature[random_integer_2:
                                      random_integer_2 + int(end / 2)]


    output_feature_one = output_feature[random_integer_1:random_integer_1 + int(
        end / 2)]
    output_feature_two = output_feature[random_integer_2:random_integer_2 + int(
        end / 2)]

    remaining_chars = 'N' * (seq_len - length)
    seq_one = seq_one + remaining_chars
    seq_two = seq_two + remaining_chars

    input_feature_one = input_feature_one + ['0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.5:0:0:0.5'] * (seq_len - length)
    output_feature_one = output_feature_one + ['0.5'] * (seq_len - length)
    input_feature_two = input_feature_two + ['0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.5:0:0:0.5'] * (seq_len - length)
    output_feature_two = output_feature_two + ['0.5'] * (seq_len - length)

    return [seq_one, seq_two], [input_feature_one, input_feature_two], [
        output_feature_one, output_feature_two]


def reverse(seq, input_feature, output_feature):
    '''
    Perform reverse on the sequence
    :param seq: orial sequenue
    :param input_feature: input data feature
    :param output_feature: output data feature
    :return: seq, input_feature, output_feature
    '''
    input_feature_list = input_feature.split(' ')
    output_feature_list = output_feature.split(' ')

    input_feature_list.reverse()
    output_feature_list.reverse()

    input_feature = f"{' '.join(input_feature_list)}"

    output_feature = f"{' '.join(output_feature_list)}"

    return seq[::-1], input_feature, output_feature


def flip(seq):
    '''
    Perform flip on the sequence
    :param seq: orial sequenu
    :return: seq
    '''
    seq_r = [rule[s] for s in seq]
    return  f"{''.join(seq_r)}"

if __name__ == '__main__':
    print(flip('ATCG'))
