import pandas as pd
import numpy as np
from joblib import load, dump
import pandas as pd
import re
from collections import defaultdict

goal_path = 'C:/Users/xiaxq/Desktop/冷热点预测课题/'

def pos_score_predict(file_path):
    model = load('C:/Users/xiaxq/Desktop/冷热点预测课题/model/RF.model')
    pre_data = pd.read_csv(file_path, sep='\t', header=None)
    pre_data_file = open(file_path,'r')
    result_data = open(file_path.replace('.txt','_addPos.txt'), 'w')

    pre_X = pre_data.iloc[1:, [0, 1]].values
    for i in pre_X:
        if i[0] == 'X':
            i[0] = 23
        if i[0] == 'Y':
            i[0] = 24
    pre_data_re = model.predict(pre_X)

    count = 0
    for line in pre_data_file:
        if count != 0:
            pos_score = pre_data_re[count-1]
            data = line.split('\t')
            headers = [
                data[0],
                data[1],
                data[2],
                data[3],
                str(pos_score),
                data[5].replace('\n',''),
            ]
            result_data.write('\t'.join(headers) +'\n')
            #result_data.write(line.replace('\n','\t') + str(pos_score) + '\n')
        count += 1

    result_data.close()

### 从原始注释结果文件提取需要的列
def extract_columns(input_file, output_file, target_headers):
    # 打开输入文件和输出文件
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        # 读取第一行（表头）
        header = infile.readline().strip()
        header_columns = header.split("\t")  # 假设文件是用制表符分隔的
        #print(len(header_columns))

        # 找到目标列的索引
        target_indices = [header_columns.index(header) for header in target_headers]

        target_indices.sort()  # 确保索引是有序的
        # 写入新的表头
        new_header = "\t".join([header_columns[i] for i in target_indices])
        outfile.write(new_header + "\n")

        # 遍历文件的每一行
        for line in infile:
            columns = line.strip().split("\t")
            #print(columns)
            ###  这一步针对annovar注释出来的数据
            if len(columns) < 159:
                length = len(columns)
                for idx in range(length, 159 + 1):
                    columns.append('-')


            # 提取目标列的内容
            new_line = "\t".join([columns[i] for i in target_indices])
            outfile.write(new_line + "\n")



## 提取原始注释文件需要的列 & 按照规定顺序排列
def process_step_1(path):
    input_file = path  # 输入文件名
    output_file = path.replace('.hg38_multianno','_extract')  # 输出文件名
    target_headers = ["Chr", "Start", "End","Ref", "Alt","Gene.refGene","GERP++_RS", "LRT_score",
                      "M-CAP_score", "MPC_score", "MVP_score", "phastCons100way_vertebrate", "phastCons17way_primate",
                      "phastCons470way_mammalian", "phyloP100way_vertebrate", "phyloP17way_primate",
                      "phyloP470way_mammalian","bStatistic_converted_rankscore","SiPhy_29way_logOdds_rankscore"
                      ]  # 目标列的标题

    # 调用函数
    extract_columns(input_file, output_file, target_headers)

    # 按照规定顺序排列
    # 读取文件
    file_path = path.replace('.hg38_multianno','_extract')
    df = pd.read_csv(file_path, sep='\t')

    # 新列顺序
    new_columns = [
        'Chr', 'Start', 'End', 'Gene.refGene', 'Ref', 'Alt', 'GERP++_RS',
        'LRT_score', 'M-CAP_score', 'MPC_score', 'MVP_score',
        'phastCons100way_vertebrate', 'phastCons17way_primate', 'phastCons470way_mammalian',
        'phyloP100way_vertebrate', 'phyloP17way_primate', 'phyloP470way_mammalian',"bStatistic_converted_rankscore","SiPhy_29way_logOdds_rankscore"
    ]
    # 重新排列列顺序
    df_reordered = df[new_columns]

    # 保存到新文件
    output_file_path = path.replace('.hg38_multianno','_reorderd')
    df_reordered.to_csv(output_file_path, sep='\t', index=False)
    print('NOTICE:process step 1 completed!')

## 去重
def process_file(input_file, output_file):
    # 用于存储每个组合（第一列+第二列）对应的行数据
    data_dict = defaultdict(lambda: {"count": 0, "values": [[] for _ in range(13)], "common_fields": None})

    # 读取文件并处理数据
    with open(input_file, "r") as infile:
        header = infile.readline().strip()  # 读取表头
        for line in infile:
            fields = line.strip().split("\t")

            # 使用第一列和第二列组合成唯一标识
            variation_key = f"{fields[0]}_{fields[1]}"

            data_dict[variation_key]["count"] += 1

            # 保存第一列、第三列和第四列的值
            if data_dict[variation_key]["common_fields"] is None:
                data_dict[variation_key]["common_fields"] = fields[0:6]

            # 处理最后11列的值（从第6列开始到第16列，索引6到16）
            for i, value in enumerate(fields[6:]):
                if value == ".":
                    data_dict[variation_key]["values"][i].append(None)
                else:
                    # 提取第一个浮点数
                    match = re.search(r"[-+]?\d*\.\d+", value)
                    if match:
                        data_dict[variation_key]["values"][i].append(float(match.group()))

    # 写入结果到输出文件
    with open(output_file, "w") as outfile:
        outfile.write(header + "\n")  # 写入表头

        for variation_key, data in data_dict.items():
            if data["count"] >= 1:  # 只处理至少有3行的组合
                # 构造输出行
                output_line = data["common_fields"]  # 保留前六列
                for values in data["values"]:  # 处理最后11列
                    if all(v is None for v in values):  # 如果所有值都是None（即'.'）
                        output_line.append(".")
                    else:
                        # 计算平均值（忽略None值）
                        non_none_values = [v for v in values if v is not None]
                        if non_none_values:  # 确保有值可计算
                            avg_value = sum(non_none_values) / len(non_none_values)
                            output_line.append(f"{avg_value:.10f}")  # 保留10位小数
                        else:
                            output_line.append(".")  # 没有有效值则保留'.'
                outfile.write("\t".join(map(str, output_line)) + "\n")



## 去重
def process_step_2(path):
    # 输入文件名和输出文件名
    input_file = path.replace('.hg38_multianno','_reorderd')  # 输入文件名
    output_file = path.replace('.hg38_multianno','_var_level')  # 输出文件名

    # 调用函数
    process_file(input_file, output_file)
    print('NOTICE:process step 2 completed!')

# 获取基因水平信息即PLI、o/e、z_score
def process_step_3(path):
    # 读取第一个文件（Excel 文件）
    gene_scores = pd.read_excel(goal_path + "Gene_feature/Gene_level_score.xlsx")

    # 创建一个字典，以基因名为键，其他列为值
    gene_dict = gene_scores.set_index('gene').to_dict(orient='index')

    # 读取第二个文件（假设为一个文本文件）
    variant_file = path.replace('.hg38_multianno','_var_level')  # 替换为你的文件名
    output_file = path.replace('.hg38_multianno','_var_gene_level') # 输出文件名

    # 打开第二个文件并逐行处理
    with open(variant_file, 'r') as infile, open(output_file, 'w') as outfile:
        # 读取文件标题并添加新的列标题
        header = infile.readline().strip()
        new_header = f"{header}\tlof.oe\tlof.pLI\tlof.z_score\tmis.oe\tmis.z_score\tsyn.oe\tsyn.z_score\n"
        outfile.write(new_header)

        # 遍历文件的每一行
        for line in infile:
            fields = line.strip().split('\t')
            fields[5:] = [field if field != "." else "0.0" for field in fields[5:]]
            gene_name = fields[3]  # 第四列是基因名

            # 检查基因名是否在字典中
            if gene_name in gene_dict:
                gene_data = gene_dict[gene_name]
                # 提取对应的值，并保留10位小数
                lof_oe = f"{gene_data['lof.oe']:.10f}"
                lof_pli = f"{gene_data['lof.pLI']:.10f}"
                lof_z_score = f"{gene_data['lof.z_score']:.10f}"
                mis_oe = f"{gene_data['mis.oe']:.10f}"
                mis_z_score = f"{gene_data['mis.z_score']:.10f}"
                syn_oe = f"{gene_data['syn.oe']:.10f}"
                syn_z_score = f"{gene_data['syn.z_score']:.10f}"
            else:
                # 如果基因名不在字典中，设置为0，并保留10位小数
                lof_oe = "0.0000000000"
                lof_pli = "0.0000000000"
                lof_z_score = "0.0000000000"
                mis_oe = "0.0000000000"
                mis_z_score = "0.0000000000"
                syn_oe = "0.0000000000"
                syn_z_score = "0.0000000000"
            fields = "\t".join(fields)
            # 将新数据添加到行末
            new_line = f"{fields.strip()}\t{lof_oe}\t{lof_pli}\t{lof_z_score}\t{mis_oe}\t{mis_z_score}\t{syn_oe}\t{syn_z_score}\n"
            outfile.write(new_line)

    print('NOTICE:process step 3 completed!')

# 将位置分数、jaRvis、冷热点分数添加上来
def process_step_4(path,pos_score_path):
    # 定义文件路径
    file1_path = pos_score_path
    file2_path = path.replace('.hg38_multianno','_var_gene_level')
    output_path = path.replace('.hg38_multianno','_var_gene_level_完整')
    # f = open('E:/冷热点预测课题_E盘分部/t.txt','w')

    # 读取第一个文件
    file1_columns = ['chr', 'start', 'end', 'ref', 'pos_score',  'jarvis_score']
    file1 = pd.read_csv(file1_path, sep='\t', header=None, names=file1_columns)
    print(f"File 1 loaded with {len(file1)} rows.")  # 调试信息：检查文件1的行数

    # 创建一个字典，用于快速查找第一个文件中的匹配行
    file1_dict = {(str(row['chr']).strip(), str(row['start']).strip()): (row['pos_score'], row['jarvis_score']) for _, row in file1.iterrows()}
    print(f"Dictionary created with {len(file1_dict)} entries.")  # 调试信息：检查字典的条目数

    # 读取第二个文件，忽略第一行（标题行）
    file2_data = pd.read_csv(file2_path, sep='\t', skiprows=1, header=None)

    # 新的标题行
    new_header = [
        "Chr", "Start", "End", "Gene","Ref","Alt", "GERP++_RS", "LRT_score",
        "M-CAP_score", "MPC_score", "MVP_score", "phastCons100way_vertebrate",
        "phastCons17way_primate", "phastCons470way_mammalian", "phyloP100way_vertebrate",
        "phyloP17way_primate", "phyloP470way_mammalian", "lof.oe", "lof.pLI", "lof.z_score",
        "mis.oe", "mis.z_score", "syn.oe", "syn.z_score","bStatistic_converted_rankscore","SiPhy_29way_logOdds_rankscore"
    ]

    # 为第二个文件添加新的列名
    file2_data.columns = new_header

    # 添加新列用于存储匹配结果
    new_columns = ['pos_score', 'jarvis_score']
    file2_data[new_columns] = None  # 初始化新列为 None

    # 遍历第二个文件，查找匹配行并填充新列
    for index, row in file2_data.iterrows():
        chr_val = str(row['Chr']).strip()
        start_val = str(row['Start']).strip()
        if (chr_val, start_val) in file1_dict:
            pos_score, jarvis_score = file1_dict[(chr_val, start_val)]
            file2_data.at[index, 'pos_score'] = pos_score
            file2_data.at[index, 'jarvis_score'] = jarvis_score
        else:
            file2_data.at[index, 'pos_score'] = '0'
            file2_data.at[index, 'jarvis_score'] = '0.5'
            # print((chr_val, start_val))

    # 调整列顺序，将 hotcold_score 移到最后
    file2_data = file2_data[new_header + ['pos_score', 'jarvis_score']]

    # 输出结果
    file2_data.to_csv(output_path, sep='\t', index=False)

    print('NOTICE:process step 4 completed!')

# 读取第一个文件的内容
def read_first_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    return lines

# 读取第二个文件的内容并转换为字典
def read_second_file(file_path):
    import pandas as pd
    df = pd.read_excel(file_path)
    # 假设第一个列是HGNC gene，最后一列是Residual Variation Intolerance Score Percentile
    return dict(zip(df.iloc[:, 0], df.iloc[:, -1] / 100))

# 更新第一个文件的内容
def update_first_file(first_file_content, second_file_dict):
    # 添加新列名
    first_file_content[0].append('Gene_Rvis')
    for row in first_file_content[1:]:
        gene = row[3]  # 假设第四列是Gene列
        if gene in second_file_dict:
            row.append(second_file_dict[gene])
        else:
            row.append(0.5)
    return first_file_content

# 将更新后的内容写回文件
def write_updated_file(file_path, content):
    with open(file_path, 'w') as f:
        for row in content:
            f.write('\t'.join(map(str, row)) + '\n')

# 加不在Annovar的基因特征
def process_step_5(path):
    # 第一个文件的路径（txt格式）
    first_file_path = path.replace('.hg38_multianno','_var_gene_level_完整')
    # 第二个文件的路径（xlsx格式）
    second_file_path = 'C:/Users/xiaxq/Desktop/冷热点预测课题/Gene_feature/RVIS.xlsx'
    # 输出文件的路径
    output_file_path = path.replace('.hg38_multianno','_var_gene_level_完整版')
    # 读取第一个文件
    first_file_content = read_first_file(first_file_path)
    # 读取第二个文件
    second_file_dict = read_second_file(second_file_path)
    # 更新第一个文件的内容
    updated_content = update_first_file(first_file_content, second_file_dict)
    # 写回更新后的文件
    write_updated_file(output_file_path, updated_content)
    print('NOTICE:process step 5 completed!')

### 重新排列annovar注释出来的文件的列顺序
def process_step_6(path):
    input_file = path.replace('.hg38_multianno','_var_gene_level_完整版')

    # 读取文件
    df = pd.read_csv(input_file, sep='\t')  # 如果是制表符分隔，使用 sep='\t'
    df = df.drop(df.columns[5], axis=1)
    # 指定新的列顺序
    new_column_order = [
        'Chr', 'Start', 'End', 'Gene','Ref',
        'GERP++_RS', 'LRT_score', 'M-CAP_score', 'MPC_score', 'MVP_score',
        'phastCons100way_vertebrate', 'phastCons17way_primate', 'phastCons470way_mammalian',
        'phyloP100way_vertebrate', 'phyloP17way_primate', 'phyloP470way_mammalian','lof.oe','lof.pLI',
        'lof.z_score',	'mis.oe',	'mis.z_score',	'syn.oe',	'syn.z_score',
        'pos_score',	'jarvis_score',	'bStatistic_converted_rankscore',
        'SiPhy_29way_logOdds_rankscore',	'Gene_Rvis'
    ]


    # 重新排列列
    df = df[new_column_order]

    df = df.rename(columns={'Ref': 'UPLOADED_ALLELE'})

    # 将结果保存到新的文件
    output_file = path.replace('.hg38_multianno','_var_gene_level_完整版_sort')
    df.to_csv(output_file, sep='\t', index=False)  # 如果是制表符分隔，使用 sep='\t'

    print('NOTICE:process step 6 completed!')

# 特征用“：”连接。
def process_step_7(path):
    # 定义文件路径
    file_path = path.replace('.hg38_multianno','_var_gene_level_完整版_sort')  # 替换为第一个文件的路径
    output_path = r'E:\冷热点预测课题_E盘分部\预测原始数据\predict_data_chr1.txt'  # 替换为输出文件的路径

    # 读取三个文件
    file1 = pd.read_csv(file_path, sep="\t")

    feature_columns = list(file1.columns[5:])  # 从 GERP++_RS 到 jarvis_score

    # 将特征分数合并为一列，用 ":" 连接
    def combine_features(row):
        return ":".join(row.astype(str))

    file1['combined_features'] = file1[feature_columns].apply(combine_features, axis=1)

    # 选择需要保留的列（假设只需要 Chr, Start, End, Gene 和合并后的特征分数列）
    final_columns = ['Chr', 'Start', 'End', 'UPLOADED_ALLELE', 'combined_features']
    final_data = file1[final_columns]

    # 输出结果，不包含列标题
    final_data.to_csv(output_path, sep="\t", index=False, header=False)

    print(f"合并后的文件已保存到 {output_path}")



def seq_process(path):
    '''
    :return: interval
    '''
    pos_score_data = pd.read_csv(
        path,
        sep='\t',header=0)
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
    # with open(r"E:\冷热点预测课题_E盘分部\test_data\test_chr18.txt", 'r') as original_data, \
    r_seq = open(r"E:\冷热点预测课题_E盘分部\test_data\test_chr18_seq.txt",'w')
    r_feature = open(r"E:\冷热点预测课题_E盘分部\test_data\test_chr18_feature.txt", 'w')
    intervals = seq_process(path)
    # file = open(path.replace('.txt','') + '_split.txt', 'w')

    df = pd.read_csv(path, sep='\t', header=0,
                     names=['chr', 'start', 'end', 'ref', 'feature'],dtype
={1: int})
    #names=['chr', 'start', 'end', 'ref', 'alt', 'pos_score', 'hotcold_score', 'jarvis_score'] names=['chr', 'start', 'ref', 'hotcold_score' , 'pos_score', 'jarvis_score']
    # format ====>  names=['chr', 'start', 'end', 'ref', 'alt', 'feature', 'hotcold_score']  feature ===> pos_score   jarvis
    for interval in intervals:
        condition = (df['start'] >= interval[0]) & (df['start'] <= interval[1])
        filtered_df = df[condition]

        chr = 'chr' + str(filtered_df['chr'].iloc[0])
        if chr == 'chr23':
            chr = 'chrX'

        seq = chr + '-' + str(interval[0]) + '-' + str(min(interval[1], interval[0] + 99)) + '\t'
        input_feature = ''
        count = 0
        for i in filtered_df['ref']:
            count += 1
            seq += str(i)
            if count == 100:
                interval[0] = interval[0] + 100
                r_seq.write(seq+'\n')
                seq = chr + '-' + str(interval[0]) + '-' + str(min(interval[1], interval[0] + 99)) + '\t'
                count = 0

        if len(seq.split('\t')[-1]) > 0:
            while len(seq.split('\t')[-1]) < 100:
                seq += 'N'
            r_seq.write(seq + '\n')

        count = 0
        for i in filtered_df['feature']:
            count += 1
            input_feature += ' ' + i
            if count == 100:
                r_feature.write(input_feature[1:] + '\n')
                input_feature = ''
                count = 0

        if len(input_feature.split(' ')) > 0:
            while len(input_feature.split(' ')) < 101:
                input_feature += ' 0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.5:0:0:0.5'
        r_feature.write(input_feature[1:] + '\n')

def predic_data_seq_split():
    """
    Process individual site prediction data into the format required by our model.

    This function reads data from a file, processes it, and writes the results to two output files:
    one for sequence data and another for feature data.
    """
    # 使用 with 语句自动管理文件资源
    with open(r"E:\冷热点预测课题_E盘分部\预测原始数据\predict_data_chr1.txt", 'r') as original_data, \
            open(r"E:\冷热点预测课题_E盘分部\预测原始数据\predict_data_chr1_seq.txt", 'w') as sequence_data, \
            open(r"E:\冷热点预测课题_E盘分部\预测原始数据\predict_data_chr1_feature.txt", 'w') as feature_data:

        sequence = ''
        feature = ''
        for count, line in enumerate(original_data):
            data = line.split('\t')
            if data[0] == 'chr':
                continue
            ref = data[3]
            all_feature = data[-1].strip()  # 使用 strip() 代替 replace('\n', '')
            chr = data[0]
            pos = int(data[1])
            # 初始化或追加序列和特征数据
            if count == 0:
                sequence += chr + '-' + str(pos) + '-' + str(pos + 99) + '\t' + ref
                feature += all_feature
            else:
                if (count + 1) % 100 != 0:
                    sequence += ref
                    if (count + 1) % 100 == 1:
                        feature += all_feature
                    else:
                        feature += ' ' + all_feature
                else:
                    sequence += ref + '\n' + chr + '-' + str(pos + 1) + '-' + str(pos + 100) + '\t'
                    feature += ' ' + all_feature + '\n'
        last_len = len(sequence.split('\n')[-1].split('\t')[-1])
        if last_len < 100:
            sequence += (100 - last_len) * 'N' + '\n'
            feature += (100 - last_len) * ' 0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.5:0:0:0.5' + '\n'


        # 写入处理后的数据
        sequence_data.write(sequence)
        feature_data.write(feature)


if __name__ == '__main__':
    for i in range(2,25):
        if i == 18:
            continue
        if i == 23:
            i = "X"
        if i == 24:
            i = "Y"
        pos_score_predict(r"E:\冷热点预测课题_E盘分部\所有外显子数据带Rvis分数\Homo_sapiens.GRCh38.exonic.chromosome" + str(i) + "_addJARIVS.txt")
        path = r"E:\冷热点预测课题_E盘分部\所有multianno数据\cold_result_chr" + str(i) + ".txt.hg38_multianno.txt"
        pos_score_path = r"E:\冷热点预测课题_E盘分部\所有外显子数据带Rvis分数\Homo_sapiens.GRCh38.exonic.chromosome" + str(i) + "_addJARIVS_addPos.txt"
        process_step_1(path)
        process_step_2(path)
        process_step_3(path)
        process_step_4(path,pos_score_path)
        process_step_5(path)
        process_step_6(path)
        process_step_7(path)




    #predic_data_seq_split()


    #seq_split(r"E:\冷热点预测课题_E盘分部\test_data\test_chr18.txt",100)