import pandas as pd
import data_augmentation
import sys
import argparse
import random


def progress_bar(finish_tasks_number, tasks_number):
    """
    进度条
    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :return:
    """

    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\r进度: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()


def seq_process(path):
    '''
    :return: interval
    '''
    pos_score_data = pd.read_csv(
        path,
        sep='\t',
        header=None)
    # file = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/seq_length.txt','w')

    pos = pos_score_data.iloc[:, 1].values
    interval = []
    interval.append([pos[0], pos[0]])

    for i in range(1, len(pos)):
        if int(pos[i]) - 1 == interval[-1][1]:
            interval[-1][1] = int(pos[i])
            continue
        else:
            interval.append([int(pos[i]), int(pos[i])])

    # for elem in interval:
    #     headers = [
    #         str(elem[0]),
    #         str(elem[1]),
    #         str(elem[1] - elem[0] + 1)]
    #     file.write(f"{'\t'.join(headers)}\n")

    return interval


def seq_split(path,seq_len):
    '''
    将单个位点数据处理成 100 长度的序列数据 训练数据
    :return:
    '''
    intervals = seq_process(path)
    file = open(path.replace('.txt','') + '_split.txt', 'w')

    df = pd.read_csv(path, sep='\t', header=None,
                     names=['chr', 'start', 'end', 'ref', 'feature', 'hotcold_score'],dtype
={1: int})
    #names=['chr', 'start', 'end', 'ref', 'alt', 'pos_score', 'hotcold_score', 'jarvis_score'] names=['chr', 'start', 'ref', 'hotcold_score' , 'pos_score', 'jarvis_score']
    # format ====>  names=['chr', 'start', 'end', 'ref', 'alt', 'feature', 'hotcold_score']  feature ===> pos_score   jarvis
    for interval in intervals:
        condition = (df['start'] >= interval[0]) & (df['start'] <= interval[1])
        filtered_df = df[condition]

        chr = 'chr' + str(filtered_df['chr'].iloc[0])
        if chr == 'chr23':
            chr = 'chrX'

        seq = ''
        input_feature = []
        output_feature = []
        for i in filtered_df['ref']:
            seq += str(i)

        for i in filtered_df['feature']:
            input_feature.append(str(i))


        for i in filtered_df['hotcold_score']:
            if i == ' ':
                i = 0.5
            output_feature.append(str(i))
        while len(seq) > seq_len:
            line = [
                chr,
                seq[0:seq_len],
                f"{' '.join(input_feature[0:seq_len])}",
                f"{' '.join(output_feature[0:seq_len])}"
            ]
            file.write(f"{'\t'.join(line)}\n")
            seq = seq[seq_len:]
            input_feature = input_feature[seq_len:]
            output_feature = output_feature[seq_len:]

        if len(seq) != 0:
            while len(seq) < seq_len:
                seq += 'N'
                input_feature.append('0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.5:0:0:0.5')
                output_feature.append('0.5')
            line = [
                chr,
                seq,
                f"{' '.join(input_feature)}",
                f"{' '.join(output_feature)}"
            ]
            file.write(f"{'\t'.join(line)}\n")

    file.close()
    data_aug(path.replace('.txt','') + '_split.txt',seq_len)


def data_aug(path,seq_len):
    '''
    数据增强
    :return:
    '''
    ori_seq_split = open(
        path, 'r')
    aug_seq_split = open(
        path.replace('.txt', '') + '_split_数据增强.txt', 'w')

    count = 0
    for line in ori_seq_split:
        count += 1
        # progress_bar(count, 20687)
        # time.sleep(0.05)
        data = line.split('\t')
        chr = data[0]
        seq = data[1]
        input_feature = data[2]
        output_feature = data[3].replace('\n', '')
        if float(output_feature.split(' ')[0]) < 0.66:  #norm 0.83
            aug_seq_split.write(line)
        elif float(output_feature.split(' ')[0]) >= 0.66 and seq.count('N') > seq_len/2:
            aug_seq_split.write(line)
        else:
            #print(line)
            aug_seq_split.write(line)
            reverse_seq, reverse_input, reverse_output = data_augmentation.reverse(
                seq, input_feature, output_feature)
            header = [
                chr,
                reverse_seq,
                reverse_input,
                reverse_output
            ]
            aug_seq_split.write(f"{'\t'.join(header)}\n")

            flip_seq = data_augmentation.flip(seq)
            header = [
                chr,
                flip_seq,
                input_feature,
                output_feature
            ]
            aug_seq_split.write(f"{'\t'.join(header)}\n")

            flip_seq = data_augmentation.flip(reverse_seq)
            header = [
                chr,
                flip_seq,
                reverse_input,
                reverse_output
            ]
            aug_seq_split.write(f"{'\t'.join(header)}\n")

            slice_seq, slice_input, slice_output = data_augmentation.slicing(
                seq, input_feature, output_feature,seq_len)
            for i in range(0, len(slice_seq)):
                header = [
                    chr,
                    slice_seq[i],
                    f"{' '.join(slice_input[i])}",
                    f"{' '.join(slice_output[i])}"
                ]
                aug_seq_split.write(f"{'\t'.join(header)}\n")

                reverse_seq, reverse_input, reverse_output = data_augmentation.reverse(slice_seq[i], f"{' '.join(slice_input[i])}",
                                                                                       f"{' '.join(slice_output[i])}")
                header = [
                    chr,
                    reverse_seq,
                    reverse_input,
                    reverse_output
                ]
                aug_seq_split.write(f"{'\t'.join(header)}\n")

                flip_seq = data_augmentation.flip(slice_seq[i])
                header = [
                    chr,
                    flip_seq,
                    f"{' '.join(slice_input[i])}",
                    f"{' '.join(slice_output[i])}"
                ]
                aug_seq_split.write(f"{'\t'.join(header)}\n")

                flip_seq = data_augmentation.flip(reverse_seq)
                header = [
                    chr,
                    flip_seq,
                    reverse_input,
                    reverse_output
                ]
                aug_seq_split.write(f"{'\t'.join(header)}\n")
    data_extract(path.replace('.txt', '') + '_split_数据增强.txt')


def data_extract(path):
    type = 'test'
    seq_split = open(
        path, 'r')
    seq_data = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输入数据_序列_100bp.txt',
        'w')
    input_feature_data = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输入数据_特征_100bp.txt', 'w')
    output_feature_data = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输出数据_冷热点分数_100bp.txt', 'w')

    # seq_data = open(
    #     'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/loss_test_序列.txt',
    #     'w')
    # input_feature_data = open(
    #     'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/loss_test_特征.txt', 'w')
    # output_feature_data = open(
    #     'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/loss_test_冷热点分数.txt', 'w')

    i = 0
    for line in seq_split :
        data = line.split('\t')
        # print(data)
        seq_data.write(data[1] + '\n')
        input_feature_data.write(data[2] + '\n')
        output_feature_data.write(data[3])
        i += 1

    seq_data.close()
    seq_split.close()
    input_feature_data.close()
    output_feature_data.close()
    print('ok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequeue process!')
    ## input file
    parser.add_argument('--seq_length', default=100,
                        type=int,
                        help='The length of input sequeue')
    args = parser.parse_args()
    path = r"E:\冷热点预测课题_E盘分部\所有数据.txt"
    seq_len = args.seq_length
    seq_split(path,seq_len)
    #data_aug(path.replace('.txt','') + '_split.txt')

