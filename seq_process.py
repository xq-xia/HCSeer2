import pandas as pd
import data_augmentation
import sys
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


def seq_process():
    '''
    :return: interval
    '''
    pos_score_data = pd.read_csv(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/所有数据.txt',
        sep='\t',
        header=None)
    # file = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/seq_length.txt','w')

    pos = pos_score_data.iloc[:, 1].values
    interval = []
    interval.append([pos[0], pos[0]])

    for i in range(1, len(pos)):
        if pos[i] - 1 == interval[-1][1]:
            interval[-1][1] = pos[i]
            continue
        else:
            interval.append([pos[i], pos[i]])

    # for elem in interval:
    #     headers = [
    #         str(elem[0]),
    #         str(elem[1]),
    #         str(elem[1] - elem[0] + 1)]
    #     file.write(f"{'\t'.join(headers)}\n")

    return interval


def seq_split():
    '''
    将单个位点数据处理成 100 长度的序列数据 训练数据
    :return:
    '''
    intervals = seq_process()
    file = open('C:/Users/xiaxq/Desktop/冷热点预测课题/feature/所有数据_split.txt', 'w')

    df = pd.read_csv('C:/Users/xiaxq/Desktop/冷热点预测课题/feature/所有数据.txt', sep='\t', header=None,
                     names=['chr', 'start', 'end', 'ref', 'feature', 'hotcold_score'])
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
            output_feature.append(str(i))
        while len(seq) > 100:
            line = [
                chr,
                seq[0:100],
                f"{' '.join(input_feature[0:100])}",
                f"{' '.join(output_feature[0:100])}"
            ]
            file.write(f"{'\t'.join(line)}\n")
            seq = seq[100:]
            input_feature = input_feature[100:]
            output_feature = output_feature[100:]

        if len(seq) != 0:
            while len(seq) < 100:
                seq += 'N'
                input_feature.append('0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.5:0.5:0:0:0.5')
                output_feature.append('0.5')
            line = [
                chr,
                seq,
                f"{' '.join(input_feature)}",
                f"{' '.join(output_feature)}"
            ]
            file.write(f"{'\t'.join(line)}\n")

    file.close()
    print('ok')


def data_aug():
    '''
    数据增强
    :return:
    '''
    ori_seq_split = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/所有数据_split.txt', 'r')
    aug_seq_split = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/数据增强_所有数据_split.txt', 'w')

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
        # print(output_feature.split(' ')[0])
        if float(output_feature.split(' ')[0]) < 0.66:  #norm 0.83
            aug_seq_split.write(line)
        elif float(output_feature.split(' ')[0]) >= 0.66 and seq.count('N') > 50:
            aug_seq_split.write(line)
        else:
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
                seq, input_feature, output_feature)
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


def data_extract():
    type = 'test'
    seq_split = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/数据增强_所有数据_split.txt', 'r')
    seq_data = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输入数据_序列.txt',
        'w')
    input_feature_data = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输入数据_特征.txt', 'w')
    output_feature_data = open(
        'C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输出数据_冷热点分数.txt', 'w')

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


def predic_data_seq_split():
    """
    Process individual site prediction data into the format required by our model.

    This function reads data from a file, processes it, and writes the results to two output files:
    one for sequence data and another for feature data.
    """
    # 使用 with 语句自动管理文件资源
    with open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_22_snv_pos_score.txt', 'r') as original_data, \
            open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_22_seq_data.txt', 'w') as sequence_data, \
            open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_22_feature_data.txt', 'w') as feature_data:

        sequence = ''
        feature = ''
        for count, line in enumerate(original_data):
            data = line.split('\t')
            ref = data[3]
            pos_score = data[-1].strip()  # 使用 strip() 代替 replace('\n', '')
            chr = data[0]
            pos = int(data[1])
            # 初始化或追加序列和特征数据
            if count == 0:
                sequence += chr + '-' + str(pos) + '-' + str(pos + 99) + '\t' + ref
                feature += pos_score
            else:
                if (count + 1) % 100 != 0:
                    sequence += ref
                    if (count + 1) % 100 == 1:
                        feature += pos_score
                    else:
                        feature += ' ' + pos_score
                else:
                    sequence += ref + '\n' + chr + '-' + str(pos + 1) + '-' + str(pos + 100) + '\t'
                    feature += ' ' + pos_score + '\n'
        last_len = len(sequence.split('\n')[-1].split('\t')[-1])
        if last_len < 100:
            sequence += (100 - last_len) * 'N' + '\n'
            feature += (100 - last_len) * ' 0.5' + '\n'


        # 写入处理后的数据
        sequence_data.write(sequence)
        feature_data.write(feature)

def var_cluster_process():
    '''
    这一步是处理非冷点非热点数据，去除与冷热点区域重叠的区间
    :return:
    '''
    import pandas as pd

    # 读取文件
    df1 = pd.read_csv('C:/Users/xiaxq/Desktop/冷热点预测课题/data/non_cold_hot_data.txt', sep='\t', header=None, names=['chr','gene','transcript', 'intervals'])
    df2 = pd.read_excel('C:/Users/xiaxq/Desktop/冷热点预测课题/data/hot_cod_interval.xlsx')

    # 准备一个空的DataFrame来存储结果
    result_df = pd.DataFrame(columns=['chr','gene','transcript', 'intervals'])

    # 遍历第一个文件的每一行
    for index, row in df1.iterrows():
        chr = row['chr']
        gene = row['gene']
        current_transcript = row['transcript']
        intervals = row['intervals'].split(';')

        # 存储剩余区间的列表
        remaining_intervals = []

        # 遍历每个区间
        for interval in intervals:
            interval1_start, interval1_end  = int(interval.split('-')[0]), int(interval.split('-')[1])
            interval1_overlap = False

            # 在第二个文件中查找相同的iso
            for _, df2_row in df2[df2['iso'] == current_transcript].iterrows():
                aa_start = df2_row['aa_start_pos']
                aa_end = df2_row['aa_end_pos']

                # 检查区间是否重叠
                if not (aa_end <= interval1_start or aa_start >= interval1_end):
                    interval1_overlap = True
                    print(current_transcript,aa_start,aa_end)
                    break

            # 如果没有重叠的区间，添加到remaining_intervals
            if not interval1_overlap:
                remaining_intervals.append(interval)

        # 如果有剩余区间，添加到结果DataFrame
        if remaining_intervals:
            result_df = result_df._append({'chr': chr, 'gene': gene, 'transcript': current_transcript, 'intervals': ';'.join(remaining_intervals)},
                                         ignore_index=True)

    # 将结果写入新文件
    result_df.to_csv('C:/Users/xiaxq/Desktop/冷热点预测课题/data/remaining_intervals.txt', sep='\t', index=False)


'''
这个方法的作用是将
1	SKI	NM_003036.4	31-37-0.625;102-118-0.3
输出为
1	SKI	NM_003036.4	31 37 0.625
1	SKI	NM_003036.4	102 118 0.3
'''
def var_cluster_process_2():
    # 定义输入和输出文件名
    input_filename = 'C:/Users/xiaxq/Desktop/冷热点预测课题/data/remaining_intervals.txt'
    output_filename = 'C:/Users/xiaxq/Desktop/冷热点预测课题/data/remaining_intervals_update.txt'

    # 打开输入文件和输出文件
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        # 逐行读取输入文件
        for line in infile:
            # 去除行尾的换行符并分割行
            parts = line.strip().split('\t')
            # 获取transcript信息和intervals字符串
            transcript_info = '\t'.join(parts[:3])
            intervals = parts[3].split(';')

            # 遍历每个区间
            for interval in intervals:
                # 分割区间字符串为起始位置、结束位置和值
                start, end, value = interval.split('-')
                # 将结果写入输出文件，每个区间一行
                outfile.write(f'{transcript_info}\t{start}\t{end}\t{value}\n')

def function_domain_process():
    function_domain_file = open('C:/Users/xiaxq/Desktop/topic-PM1-tep-file/functional_domains_hg38.bed.txt','r')
    result = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data/function_domain.txt','w')

    for line in function_domain_file:
        data = line.split('\t')
        if data[3].split('|')[-1] == '0' and data[3].split('|')[-2] == '0':
            continue

        if data[3].split('|')[-1] == '0':
            data[3].split('|')[-1] = '1'
        if data[3].split('|')[-2] == '0':
            data[3].split('|')[-2] = '1'
        header = [
            data[0],
            data[1],
            data[2],
            str(int(data[2]) - int(data[1])),
            data[3].split('|')[-4],
            data[3].split('|')[-2],
            data[3].split('|')[-1]
        ]
        result.write(f"{'\t'.join(header)}\n")

    function_domain_file.close()
    result.close()


'''
这个方法作用是将冷热点分数转换为到1之间
'''
def normzation(file_path):
    f = open(file_path, 'r')
    r = open(file_path.replace('.txt','') + '_norm.txt','w' )

    for line in f:
        data = line.split('\t')
        data[-1] = str((float(data[-1].strip()) + 1) / 2)
        r.write(f"{'\t'.join(data)}\n")
    f.close()
    r.close()



if __name__ == '__main__':
    seq_split()
    data_aug()
    data_extract()
    #var_cluster_process_2()

    # hc = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/seq_hot_addjarvis_norm.txt','r')
    # fun = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/seq_function_addjarvis_norm.txt','r')
    # var = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/seq_invar_addjarvis_norm.txt', 'r')
    #
    # fin = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/fin_seq_feature_data_addjarvis_norm.txt', 'w')
    #
    # for l in hc:
    #     fin.write(l)
    # for l in fun:
    #     fin.write(l)
    #
    # for l in var:
    #     fin.write(l)
    #
    # hc.close()
    # fun.close()
    # var.close()
    # fin.close()


    '''
    normzation('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/All_inVarspot_file_fin_pos_score_update_0.005.txt')
    normzation('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/function_domain_0.005_pos_score.txt')
    normzation('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/hot_cold_Pos_Score_update_annotation_result_0.005.txt')
    '''


    #function_domain_process()
    # f = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/function_domain_interval.txt', 'r')
    # for l in f:
    #     data = l.split('\t')
    #     print(data)
    #predic_data_seq_split()