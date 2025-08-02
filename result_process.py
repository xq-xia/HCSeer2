import pandas as pd
import data_augmentation
import sys
import random

def extract():
    key_word = {'A':'C','C':'A','G':'T','T':'G'}
    hot = open(r'E:\冷热点预测课题_E盘分部\predict_result\hot_result_all.txt', 'w')
    cold = open(r'E:\冷热点预测课题_E盘分部\predict_result\cold_result_all.txt', 'w')
    mid = open(r'E:\冷热点预测课题_E盘分部\predict_result\mid_result_all.txt', 'w')
    for i in range(1,25):
        if i == 23:
            i = "X"
        if i == 24:
            i = "Y"
        path = r"E:\冷热点预测课题_E盘分部\predict_result\predict_result_chr" + str(i) + ".txt"
        f = open(path,'r')
        for line in f:
            data = line.split('\t')
            if float(line.split('\t')[-1].replace('\n','')) >= 0.66 - 0.015:
                headers = [
                    data[0],
                    data[1],
                    data[2],
                    data[3],
                    key_word[data[3]],
                    data[4]
                ]
                hot.write('\t'.join(headers))
            if float(line.split('\t')[-1].replace('\n','')) <= 0.33:
                headers = [
                    data[0],
                    data[1],
                    data[2],
                    data[3],
                    key_word[data[3]],
                    data[4]
                ]
                cold.write('\t'.join(headers))
            if float(line.split('\t')[-1].replace('\n','')) > 0.33 and float(line.split('\t')[-1].replace('\n','')) < 0.66 - 0.015:
                headers = [
                    data[0],
                    data[1],
                    data[2],
                    data[3],
                    key_word[data[3]],
                    data[4]
                ]
                mid.write('\t'.join(headers))
        print('chr' + str(i)+'已完成')
    f.close()
    hot.close()
    cold.close()
    mid.close()



import pandas as pd
from itertools import groupby


def process_hotspots(input_file, output_file):
    # 读取文件，假设是以制表符分隔
    df = pd.read_csv(input_file, sep='\t', header=None,
                     names=['chr', 'start', 'end', 'ref', 'alt', 'score',
                            'gene', 'transcript', 'exon', 'aa_pos'])

    results = []

    # 按基因和转录本分组
    grouped = df.groupby(['gene', 'transcript'])

    for (gene, transcript), group in grouped:
        # 按氨基酸位置排序
        group = group.sort_values('aa_pos')

        # 找出连续的氨基酸位置区间
        aa_positions = group['aa_pos'].unique()
        ranges = []

        # 使用groupby来分组连续的氨基酸位置
        for k, g in groupby(enumerate(aa_positions), lambda x: x[0] - x[1]):
            consecutive = list(g)
            start = consecutive[0][1]
            end = consecutive[-1][1]
            ranges.append((start, end))

        # 对每个连续区间计算平均分数
        for start, end in ranges:
            mask = (group['aa_pos'] >= start) & (group['aa_pos'] <= end)
            subset = group[mask]
            avg_score = subset['score'].mean()

            # 获取染色体信息
            chr_info = subset['chr'].iloc[0]

            results.append([
                chr_info, gene, transcript,
                str(start), str(end),
                f"{avg_score:.9f}".rstrip('0').rstrip('.')  # 格式化分数，去除多余的0
            ])

    # 写入输出文件
    with open(output_file, 'w') as f:
        f.write('chr'+'\t'+'gene'+'\t'+'iso'+'\t'+'aa_start_pos'+'\t'+'aa_end_pos'+'\t'+'hot_cold_score'+'\n')
        for row in results:
            f.write('\t'.join(row) + '\n')






if __name__ == '__main__':
    # extract()
    # 使用示例

    input_file = r"E:\冷热点预测课题_E盘分部\predict_result\hot_result_all_get_maneiso.txt"
    output_file = r"E:\冷热点预测课题_E盘分部\predict_result\hotspot_regions_all_results.txt"
    process_hotspots(input_file, output_file)

    print(f"结果已保存到: {output_file}")
