import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import numpy as np
from joblib import load, dump


def data_process():
    '''
    :return:
    '''

    import pandas as pd
    file_path = r'C:\Users\xiaxq\Desktop\冷热点预测课题\data\RF_pos_score1.xlsx'
    df = pd.read_excel(file_path, engine='openpyxl')
    selected_columns = df[['chr', 'start', 'pos_score']]
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
    input_file_path = r"E:\冷热点预测课题_E盘分部\data_set\pos_score_train_data.txt"
    output_file_path = r"E:\冷热点预测课题_E盘分部\data_set\modified_pos_score_train_data.txt"
    modification_rules = {
        '1': '1',
        '0.75': '0.5',
        '0.5': '0',
        '0.25': '-0.5',
        '0': '-1'
    }
    with open(input_file_path, 'r', encoding='utf-8') as file, open(output_file_path, 'w',
                                                                    encoding='utf-8') as output_file:
        for line in file:
            parts = line.strip().split()
            last_value = parts[-1]
            new_value = modification_rules.get(last_value, last_value) 
            parts[-1] = new_value
            new_line = '\t'.join(parts) + '\n'
            output_file.write(new_line)

def compute_pos_score_train():
        data = pd.read_csv(r"E:\冷热点预测课题_E盘分部\data_set\modified_pos_score_train_data.txt",sep='\t', header=None)
        #pre_data = pd.read_csv('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_9_snv.txt', sep='\t', header=None)
        #pre_data_file = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_9_snv.txt','r')
        #result_data = open('C:/Users/xiaxq/Desktop/冷热点预测课题/data_set/chr_9_snv_pos_score.txt', 'w')
        X = data.iloc[:, [0, 1]].values  
        y = data.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        dump(model, 'C:/Users/xiaxq/Desktop/冷热点预测课题/model/RF.model')

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


if __name__ == '__main__':
    pos_score_predict()
    #compute_pos_score_train()

