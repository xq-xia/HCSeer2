import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import numpy as np
from joblib import load, dump


def data_process():
    '''
    处理clinvar中错义变异，设置2星及以上PLP、BLB变异位置分数为1和-1，
    所有2星以下PLP、BLB变异的位置分数为0.5及-0.5，所有VUS及CON变异位置分数设置为0
    :return:
    '''

    import pandas as pd

    # 文件路径
    file_path = r'C:\Users\xiaxq\Desktop\冷热点预测课题\data\RF_pos_score1.xlsx'

    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(file_path, engine='openpyxl')

    # 选择第一列、第二列和第四列
    # 假设列的名称分别是 'Column1', 'Column2', 'Column4'
    # 如果你不知道列的名称，可以使用 df.columns 来查看所有列的名称
    selected_columns = df[['chr', 'start', 'pos_score']]

    # 将选择的列保存到 txt 文件中，列之间用制表符分隔
    output_file_path = r'E:\冷热点预测课题_E盘分部\data_set\pos_score_train_data.txt'
    selected_columns.to_csv(output_file_path, sep='\t', index=False, header=False)


    with open(r"E:\冷热点预测课题_E盘分部\data_set\clinvar_20240407_hg38_missense.vcf",'r') as clinvar_missense_file, \
            open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/pos_score_train_data.txt', 'a') as pos_score_train_data :
        for line in clinvar_missense_file:
            data = line.split('\t')
            chr = data[0]
            if chr == 'X':
                chr = '23'
            if chr == 'Y':
                chr = '24'

            pos = data[1]
            pos_score = 0
            info = data[4]
            if "CLNSIG=Pathogenic;" in info or "CLNSIG=Likely_pathogenic;" in info or "CLNSIG=Pathogenic/Likely_pathogenic;" in info:
                if "CLNREVSTAT=criteria_provided,_multiple_submitters" in info or "reviewed_by_expert_panel" in info or "CLNREVSTAT=practice_guideline" in info:
                    pos_score = 1
                else:
                    pos_score = 0.5

            elif "CLNSIG=Likely_benign;" in info or "CLNSIG=Benign;" in info or "CLNSIG=Benign/Likely_benign;" in info:
                if "CLNREVSTAT=criteria_provided,_multiple_submitters" in info or "reviewed_by_expert_panel" in info or "CLNREVSTAT=practice_guideline" in info:
                    pos_score = -1
                else:
                    pos_score = -0.5
            else:
                pos_score = 0
            header = [
                chr,
                pos,
                str(pos_score)
            ]
            pos_score_train_data.write('\t'.join(header) + '\n')

def proess():
    # 定义文件路径
    input_file_path = r"E:\冷热点预测课题_E盘分部\data_set\pos_score_train_data.txt"
    output_file_path = r"E:\冷热点预测课题_E盘分部\data_set\modified_pos_score_train_data.txt"

    # 定义修改规则
    modification_rules = {
        '1': '1',
        '0.75': '0.5',
        '0.5': '0',
        '0.25': '-0.5',
        '0': '-1'
    }

    # 读取文件并进行修改
    with open(input_file_path, 'r', encoding='utf-8') as file, open(output_file_path, 'w',
                                                                    encoding='utf-8') as output_file:
        for line in file:
            # 分割每一行的数据
            parts = line.strip().split()
            # 获取最后一列的值
            last_value = parts[-1]
            # 根据修改规则修改值
            new_value = modification_rules.get(last_value, last_value)  # 如果没有匹配的规则，保持原值不变
            # 重新组合这一行的数据
            parts[-1] = new_value
            new_line = '\t'.join(parts) + '\n'
            # 写入新的文件
            output_file.write(new_line)

    print("文件修改完成，修改后的内容已保存到:", output_file_path)

def compute_pos_score_train():
        # 加载数据
        data = pd.read_csv(r"E:\冷热点预测课题_E盘分部\data_set\modified_pos_score_train_data.txt",sep='\t', header=None)
        #pre_data = pd.read_csv('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_9_snv.txt', sep='\t', header=None)
        #pre_data_file = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_9_snv.txt','r')
        #result_data = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_9_snv_pos_score.txt', 'w')

        # 特征和标签
        X = data.iloc[:, [0, 1]].values  # 将特征转换为二维数组
        y = data.iloc[:, 2].values


        #pre_X = pre_data.iloc[:, [0, 1]].values
        # pre_X = []
        #
        #
        # for i in range(0,100):
        #     pre_X.append([])
        #     pre_X[-1].append(20)
        #     pre_X[-1].append(57484595 + i*1000)
        #数据集分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #print(X_test)
        # 模型训练
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        dump(model, 'C:/Users/xiaxq/Desktop/冷热点预测课题/model/RF.model')
        # 模型预测
        # print('X',X_test)
        # y_pred = model.predict(X_test)
        # y_pred_proba = model.predict_proba(X_test)
        #
        # # 打印预测概率
        # print(y_pred_proba[1])
        # # 模型评估
        # accuracy = accuracy_score(y_test, y_pred)
        # # for i in range(0,len(y_pred)):
        # #     print(y_pred[i],y_test[i])
        # print(f'模型准确率: {accuracy:.2f}')

        #pre_data_re = model.predict_proba(pre_X)

        #
        #
        # count = 0
        # for line in pre_data_file:
        #     pos_score = pre_data_re[count][1]
        #     result_data.write(line.replace('\n','\t') + str(pos_score) + '\n')
        #     count += 1
        #
        # result_data.close()
def pos_score_predict():
    model = load('C:/Users/xiaxq/Desktop/冷热点预测课题/model/RF.model')
    pre_data = pd.read_csv(r"E:\冷热点预测课题_E盘分部\ClinGen PM1数据\ClinGen_PM1_Set_addJARIVS.txt", sep='\t', header=None)
    pre_data_file = open(r"E:\冷热点预测课题_E盘分部\ClinGen PM1数据\ClinGen_PM1_Set_addJARIVS.txt",'r')
    result_data = open(r"E:\冷热点预测课题_E盘分部\ClinGen PM1数据\ClinGen_PM1_Set_addJARIVS_addPos.txt", 'w')

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
                data[6].replace('\n',''),
                data[5]
            ]
            result_data.write('\t'.join(headers) +'\n')
            #result_data.write(line.replace('\n','\t') + str(pos_score) + '\n')
        count += 1

    result_data.close()

def t_pos_score_predict():
    path = 'C:/Users/xiaxq/Desktop/冷热点预测课题/'
    model = load('C:/Users/xiaxq/Desktop/冷热点预测课题/model/RF.model')
    pre_data = pd.read_csv(path + "test_data/chr1_test_data_去重_排序_添加基因水平信息.txt", sep='\t', header=None)
    pre_data_file = open(path + "test_data/chr1_test_data_去重_排序_添加基因水平信息.txt",'r')
    result_data = open(path + "test_data/chr1_test_data_去重_排序_添加基因水平信息_位置分数.txt", 'w')

    pre_X = pre_data.iloc[1:, [0, 1]].values
    for idx in pre_X:
        idx[0] = idx[0].replace('chr','')
    pre_data_re = model.predict(pre_X)

    count = -1
    for line in pre_data_file:
        if count == -1:
            result_data.write(line.replace('\n', '\t') + 'pos_score' + '\n')
        else:
            pos_score = pre_data_re[count]
            # headers = [
            #     line.replace('\n','')
            #     str(pos_score)
            # ]
            # result_data.write(f"{'\t'.join(headers)}")
            result_data.write(line.replace('\n', '\t') + str(pos_score) + '\n')
        count += 1

    result_data.close()



if __name__ == '__main__':
    '''
    f = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/pos_score_train_data.txt', 'r')
    p = 0
    b = 0
    for l in f:
        if l.split('\t')[-1].strip() == '1':
            p += 1
        if l.split('\t')[-1].strip() == '0':
            b += 1

    f.close()
    print(p,b)
    '''
    pos_score_predict()
    #compute_pos_score_train()
