import pandas as pd
import re
from collections import defaultdict

goal_path = 'C:/Users/xiaxq/Desktop/冷热点预测课题/'

def extract_columns(input_file, output_file, target_headers):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        header = infile.readline().strip()
        header_columns = header.split("\t") 
        #print(len(header_columns))
        target_indices = [header_columns.index(header) for header in target_headers]

        target_indices.sort()  
        new_header = "\t".join([header_columns[i] for i in target_indices])
        outfile.write(new_header + "\n")

        for line in infile:
            columns = line.strip().split("\t")
            #print(columns)
            if len(columns) < 159:
                length = len(columns)
                for idx in range(length, 159 + 1):
                    columns.append('-')

            new_line = "\t".join([columns[i] for i in target_indices])
            outfile.write(new_line + "\n")

def process_step_1(path):
    input_file = path  
    output_file = path.replace('.hg38_multianno','_extract')  
    target_headers = ["Chr", "Start", "End","Ref", "Alt","Gene.refGene","GERP++_RS", "LRT_score",
                      "M-CAP_score", "MPC_score", "MVP_score", "phastCons100way_vertebrate", "phastCons17way_primate",
                      "phastCons470way_mammalian", "phyloP100way_vertebrate", "phyloP17way_primate",
                      "phyloP470way_mammalian","bStatistic_converted_rankscore","SiPhy_29way_logOdds_rankscore"
                      ]  
    extract_columns(input_file, output_file, target_headers)
    file_path = path.replace('.hg38_multianno','_extract')
    df = pd.read_csv(file_path, sep='\t')

    new_columns = [
        'Chr', 'Start', 'End', 'Gene.refGene', 'Ref', 'Alt', 'GERP++_RS',
        'LRT_score', 'M-CAP_score', 'MPC_score', 'MVP_score',
        'phastCons100way_vertebrate', 'phastCons17way_primate', 'phastCons470way_mammalian',
        'phyloP100way_vertebrate', 'phyloP17way_primate', 'phyloP470way_mammalian',"bStatistic_converted_rankscore","SiPhy_29way_logOdds_rankscore"
    ]
    df_reordered = df[new_columns]
    output_file_path = path.replace('.hg38_multianno','_reorderd')
    df_reordered.to_csv(output_file_path, sep='\t', index=False)
    print('NOTICE:process step 1 completed!')

def process_file(input_file, output_file):
    data_dict = defaultdict(lambda: {"count": 0, "values": [[] for _ in range(13)], "common_fields": None})
    with open(input_file, "r") as infile:
        header = infile.readline().strip()
        for line in infile:
            fields = line.strip().split("\t")
            variation_key = f"{fields[0]}_{fields[1]}"
            data_dict[variation_key]["count"] += 1
            if data_dict[variation_key]["common_fields"] is None:
                data_dict[variation_key]["common_fields"] = fields[0:6]
            for i, value in enumerate(fields[6:]):
                if value == ".":
                    data_dict[variation_key]["values"][i].append(None)
                else:
                    match = re.search(r"[-+]?\d*\.\d+", value)
                    if match:
                        data_dict[variation_key]["values"][i].append(float(match.group()))

    with open(output_file, "w") as outfile:
        outfile.write(header + "\n")

        for variation_key, data in data_dict.items():
            if data["count"] >= 1: 
                output_line = data["common_fields"]  
                for values in data["values"]:  
                    if all(v is None for v in values):  
                        output_line.append(".")
                    else:
                        non_none_values = [v for v in values if v is not None]
                        if non_none_values:  
                            avg_value = sum(non_none_values) / len(non_none_values)
                            output_line.append(f"{avg_value:.10f}")  
                        else:
                            output_line.append(".")  
                outfile.write("\t".join(map(str, output_line)) + "\n")



def process_step_2(path):
    input_file = path.replace('.hg38_multianno','_reorderd')  
    output_file = path.replace('.hg38_multianno','_var_level')  
    process_file(input_file, output_file)
    print('NOTICE:process step 2 completed!')

# get gnen level info. PLI、o/e、z_score
def process_step_3(path):
    gene_scores = pd.read_excel(goal_path + "Gene_feature/Gene_level_score.xlsx")
    gene_dict = gene_scores.set_index('gene').to_dict(orient='index')
    variant_file = path.replace('.hg38_multianno','_var_level')  
    output_file = path.replace('.hg38_multianno','_var_gene_level') 
    with open(variant_file, 'r') as infile, open(output_file, 'w') as outfile:
        header = infile.readline().strip()
        new_header = f"{header}\tlof.oe\tlof.pLI\tlof.z_score\tmis.oe\tmis.z_score\tsyn.oe\tsyn.z_score\n"
        outfile.write(new_header)
        for line in infile:
            fields = line.strip().split('\t')
            fields[5:] = [field if field != "." else "0.0" for field in fields[5:]]
            gene_name = fields[3]  
            if gene_name in gene_dict:
                gene_data = gene_dict[gene_name]
                lof_oe = f"{gene_data['lof.oe']:.10f}"
                lof_pli = f"{gene_data['lof.pLI']:.10f}"
                lof_z_score = f"{gene_data['lof.z_score']:.10f}"
                mis_oe = f"{gene_data['mis.oe']:.10f}"
                mis_z_score = f"{gene_data['mis.z_score']:.10f}"
                syn_oe = f"{gene_data['syn.oe']:.10f}"
                syn_z_score = f"{gene_data['syn.z_score']:.10f}"
            else:
                lof_oe = "0.0000000000"
                lof_pli = "0.0000000000"
                lof_z_score = "0.0000000000"
                mis_oe = "0.0000000000"
                mis_z_score = "0.0000000000"
                syn_oe = "0.0000000000"
                syn_z_score = "0.0000000000"
            fields = "\t".join(fields)
            new_line = f"{fields.strip()}\t{lof_oe}\t{lof_pli}\t{lof_z_score}\t{mis_oe}\t{mis_z_score}\t{syn_oe}\t{syn_z_score}\n"
            outfile.write(new_line)

    print('NOTICE:process step 3 completed!')
def process_step_4(path):
    file1_path = r"E:\冷热点预测课题_E盘分部\第二次更新的冷热点数据\hot_cold_addScore_addJARIVS_addPos.txt"
    file2_path = path.replace('.hg38_multianno','_var_gene_level')
    output_path = path.replace('.hg38_multianno','_var_gene_level_完整')
    # f = open('E:/冷热点预测课题_E盘分部/t.txt','w')

    file1_columns = ['chr', 'start', 'end', 'ref', 'pos_score',  'jarvis_score', 'hotcold_score']
    file1 = pd.read_csv(file1_path, sep='\t', header=None, names=file1_columns)
    print(f"File 1 loaded with {len(file1)} rows.") 
    file1_dict = {(str(row['chr']).strip(), str(row['start']).strip()): (row['pos_score'], row['hotcold_score'], row['jarvis_score']) for _, row in file1.iterrows()}
    print(f"Dictionary created with {len(file1_dict)} entries.") 
    file2_data = pd.read_csv(file2_path, sep='\t', skiprows=1, header=None)
    new_header = [
        "Chr", "Start", "End", "Gene","Ref","Alt", "GERP++_RS", "LRT_score",
        "M-CAP_score", "MPC_score", "MVP_score", "phastCons100way_vertebrate",
        "phastCons17way_primate", "phastCons470way_mammalian", "phyloP100way_vertebrate",
        "phyloP17way_primate", "phyloP470way_mammalian", "lof.oe", "lof.pLI", "lof.z_score",
        "mis.oe", "mis.z_score", "syn.oe", "syn.z_score","bStatistic_converted_rankscore","SiPhy_29way_logOdds_rankscore"
    ]

    file2_data.columns = new_header
    new_columns = ['pos_score', 'jarvis_score', 'hotcold_score']
    file2_data[new_columns] = None 

    for index, row in file2_data.iterrows():
        chr_val = str(row['Chr']).strip()
        start_val = str(row['Start']).strip()
        if (chr_val, start_val) in file1_dict:
            pos_score, hotcold_score, jarvis_score = file1_dict[(chr_val, start_val)]
            file2_data.at[index, 'pos_score'] = pos_score
            file2_data.at[index, 'jarvis_score'] = jarvis_score
            file2_data.at[index, 'hotcold_score'] = hotcold_score
        else:
            file2_data.at[index, 'pos_score'] = '0'
            file2_data.at[index, 'jarvis_score'] = '0.5'
            file2_data.at[index, 'hotcold_score'] = '0.5'
            # print((chr_val, start_val))

    file2_data = file2_data[new_header + ['pos_score', 'jarvis_score', 'hotcold_score']]

    file2_data.to_csv(output_path, sep='\t', index=False)

    print('NOTICE:process step 4 completed!')
def read_first_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    return lines

def read_second_file(file_path):
    import pandas as pd
    df = pd.read_excel(file_path)
    return dict(zip(df.iloc[:, 0], df.iloc[:, -1] / 100))

def update_first_file(first_file_content, second_file_dict):
    first_file_content[0].append('Gene_Rvis')
    for row in first_file_content[1:]:
        gene = row[3] 
        if gene in second_file_dict:
            row.append(second_file_dict[gene])
        else:
            row.append(0.5)
    return first_file_content
def write_updated_file(file_path, content):
    with open(file_path, 'w') as f:
        for row in content:
            f.write('\t'.join(map(str, row)) + '\n')

def process_step_5(path):
    first_file_path = path.replace('.hg38_multianno','_var_gene_level_完整')
    second_file_path = 'C:/Users/xiaxq/Desktop/冷热点预测课题/Gene_feature/RVIS.xlsx'
    output_file_path = path.replace('.hg38_multianno','_var_gene_level_完整版')
    first_file_content = read_first_file(first_file_path)
    second_file_dict = read_second_file(second_file_path)
    updated_content = update_first_file(first_file_content, second_file_dict)
    write_updated_file(output_file_path, updated_content)
    print('NOTICE:process step 5 completed!')

def process_step_6(path):
    input_file = path.replace('.hg38_multianno','_var_gene_level_完整版')
    df = pd.read_csv(input_file, sep='\t')  
    df = df.drop(df.columns[5], axis=1)
    new_column_order = [
        'Chr', 'Start', 'End', 'Gene','Ref',
        'GERP++_RS', 'LRT_score', 'M-CAP_score', 'MPC_score', 'MVP_score',
        'phastCons100way_vertebrate', 'phastCons17way_primate', 'phastCons470way_mammalian',
        'phyloP100way_vertebrate', 'phyloP17way_primate', 'phyloP470way_mammalian','lof.oe','lof.pLI',
        'lof.z_score',	'mis.oe',	'mis.z_score',	'syn.oe',	'syn.z_score',
        'pos_score',	'jarvis_score',	'hotcold_score',	'bStatistic_converted_rankscore',
        'SiPhy_29way_logOdds_rankscore',	'Gene_Rvis'
    ]

    df = df[new_column_order]

    df = df.rename(columns={'Ref': 'UPLOADED_ALLELE'})
    output_file = path.replace('.hg38_multianno','_var_gene_level_完整版_sort')
    df.to_csv(output_file, sep='\t', index=False)  
    print('NOTICE:process step 6 completed!')

def process_step_7(path):
    file4_path = path.replace('.hg38_multianno','_var_gene_level_完整版_sort')  
    file1_path = r"E:\冷热点预测课题_E盘分部\第二次更新的冷热点数据\annovar_source_data\hot_cold_var_gene_level_完整版_sort.txt"
    file2_path = r"C:\Users\xiaxq\Desktop\冷热点预测课题\feature\updated_posScore_All_inVar_info_加基因特征.txt" 
    file3_path = r"C:\Users\xiaxq\Desktop\冷热点预测课题\feature\updated_posScore_function_domain_info_加基因特征.txt"  
    output_path = 'E:/冷热点预测课题_E盘分部/所有数据.txt'  
    file1 = pd.read_csv(file1_path, sep="\t")
    file2 = pd.read_csv(file2_path, sep="\t")
    file3 = pd.read_csv(file3_path, sep="\t")
    file4 = pd.read_csv(file4_path, sep="\t")
    merged_data = pd.concat([file1, file2, file3,file4], ignore_index=True)
    feature_columns = list(merged_data.columns[5:25]) + list(merged_data.columns[26:])  # for GERP++_RS to jarvis_score
    def combine_features(row):
        return ":".join(row.astype(str))

    merged_data['combined_features'] = merged_data[feature_columns].apply(combine_features, axis=1)
    final_columns = ['Chr', 'Start', 'End', 'UPLOADED_ALLELE', 'combined_features', 'hotcold_score']
    final_data = merged_data[final_columns]
    final_data.to_csv(output_path, sep="\t", index=False, header=False)

    print(f"合并后的文件已保存到 {output_path}")


if __name__ == '__main__':
    path = r"E:\冷热点预测课题_E盘分部\ClinGen PM1数据\clingen_pm1.hg38_multianno.txt"
    process_step_1(path)
    process_step_2(path)
    process_step_3(path)
    process_step_4(path)
    process_step_5(path)
    process_step_6(path)
    process_step_7(path)


