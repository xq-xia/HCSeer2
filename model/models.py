import torch
import torch.nn as nn
import model.blocks as blocks
import torch.nn.functional as F


'''
Code reference:
@article{He2024.11.23.translatomer,
    title = {Deep learning prediction of ribosome profiling with Translatomer reveals translational regulation and interprets disease variants},
    author = {Jialin He and Lei Xiong, Shaohui Shi and Chengyu Li and Kexuan Chen and Qianchen Fang and Jiuhong Nan and Ke Ding, Yuanhui Mao and Carles A. Boix and Xinyang Hu and Manolis Kellis and Jingyun Li, Xushen Xiong},
    year = {2024},
    doi = {10.1038/s42256-024-00915-6},
    publisher = {},
    url = {https://doi.org/10.1038/s42256-024-00915-6},
    journal = {Nature Machine Intelligence}
}
'''

class TransModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden=12, record_attn=False, seq_len=100):
        super(TransModel, self).__init__()
        input_number = 5 + num_genomic_features
        print('Initializing TransModel')
        print(input_number)

        self.norm = nn.BatchNorm1d(input_number)
        self.conv1 = nn.Sequential(
            # nn.Conv1d(input_number, 10, 11, 10, 5),
            # nn.Conv1d(1, 512, 129, 128, 64),
            nn.Conv1d(input_number, 10, int(seq_len / 10 + 1), int(seq_len / 10 + 1), 5),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        self.attn = blocks.AttnModule(hidden=mid_hidden, record_attn=record_attn, inpu_dim=10)
        self.conv2 = nn.Conv1d(10, 1, 3, 1, 1)
        self.Linear1 = nn.Linear(in_features=10, out_features=seq_len)
        self.record_attn = record_attn
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):

        x = x.transpose(1, 2).contiguous().float()
        x = self.conv1(x)
        x = x.transpose(1, 2).contiguous().float()
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x = self.attn(x)
        x = self.dropout(x)
        x = x.transpose(1, 2).contiguous().float()
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.Linear1(x).squeeze(1)
        x = F.relu(x)
        if self.record_attn:
            return x, attn_weights
        else:
            return x

