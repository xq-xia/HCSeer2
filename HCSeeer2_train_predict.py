import sys
import torch
import argparse
import numpy as np
import os
import json
import pandas as pd
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torchmetrics import Metric

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import default_collate

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
import pytorch_lightning.callbacks as callbacks

from typing import Optional
import model.models as models

from tensor_loader import TensorLoader

import time
import random
import re

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEQ_LEN = 100
TARGET_LEN = 100

log = open('predict_result/log_200.txt','a+')
#pc_f = open('predict_result/pc.txt','a+')
# Correlation computation along positions from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/metrics.py
class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    full_state_update: bool = False
    higher_is_better: Optional[bool] = True

    def __init__(self, n_channels: int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_dims = (0, 1)
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (self.product
                      - true_mean * self.pred
                      - pred_mean * self.true
                      + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence, input_features, output_features, seq_len, target_length, use_aug=True):
        self.target_length = target_length
        self.seq_len = seq_len

        self.sequence = sequence
        # print('=============================')
        # print('len(self.sequence)',len(self.sequence))
        self.input_features = input_features
        self.output_features = output_features

        self.use_aug = use_aug

    @staticmethod
    def one_hot_encode(sequence):
        en_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
        en_seq = [en_dict[ch] for ch in sequence]
        np_seq = np.array(en_seq, dtype=int)
        seq_emb = np.zeros((len(np_seq), 5))
        seq_emb[np.arange(len(np_seq)), np_seq] = 1
        return seq_emb.astype(np.float32)

    def __len__(self):
        return len(self.sequence)

    def reverse(self, seq, input_features, output_features, strand):
        '''
        Reverse sequence and matrix
        '''
        if strand == '-':
            seq_r = np.flip(seq, 0).copy()  # n x 5 shape
            input_features_r = torch.flip(input_features, dims=[0])
            output_features_r = torch.flip(output_features, dims=[0])  # n
            # Complementary sequence
            seq_r = self.complement(seq_r)
        else:
            seq_r = seq
            input_features_r = input_features
            output_features_r = output_features
        return seq_r, input_features_r, output_features_r

    def complement(self, seq):
        '''
        Complimentary sequence
        '''
        seq_comp = np.concatenate([seq[:, 1:2],
                                   seq[:, 0:1],
                                   seq[:, 3:4],
                                   seq[:, 2:3],
                                   seq[:, 4:5]], axis=1)
        return seq_comp

    def __getitem__(self, idx):
        sequence = self.sequence[idx]
        count_N = len(sequence) - sequence.count('N')
        sequence_one_hot = self.one_hot_encode(sequence)
        input_features = self.input_features[0][idx]
        if len(self.output_features) > 0:
            output_features = self.output_features[0][idx]
        else:
            output_features = []
        # if self.use_aug:
        #   sequence_one_hot, input_features, output_features = self.reverse(sequence_one_hot, input_features, output_features, strand)
        feature_number = 23
        input_features_list = []
        for feature_length in range(0, feature_number):
            input_features_list.append([])

        for elem in input_features:
            elem = elem.split(':')
            #elem.reverse()
            for feature_length in range(0, feature_number):
            #index = [0,1,2,3,4,5,6,7,9,10,11,18,19]
            #i = 0
            #for feature_length in index:
                try:
                    input_features_list[feature_length].append(float(elem[feature_length]))
                except:
                    print(input_features)

                #input_features_list[i].append(float(elem[feature_length]))
                #i += 1

        for feature_length in range(0, feature_number):
            input_features_list[feature_length] = torch.Tensor(input_features_list[feature_length])



        # print(input_features)
        #print(input_features_2)
        return {
            'count_N': count_N,
            'sequence': sequence_one_hot,
            'input_features': input_features_list,
            'output_features': output_features,
        }


class DataModule(LightningDataModule):
    def __init__(
            self,
            sequence: str = None,
            sequence_val: str = None,
            input_file: list = [],
            input_file_val: list = [],
            output_file: list = [],
            output_file_val: list = [],
            predict_seq_file: list = [],
            predict_file: list = [],
            seq_len: int = 100,
            target_length: int = 100,
            batch_size: int = 32,
            eval_batch_size: int = None,
            num_workers: int = 2,
            pin_memory: bool = True,
            **kwargs
    ):
        super().__init__()
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sequence = sequence
        self.sequence_val = sequence_val
        self.input_features = input_file
        self.input_features_val = input_file_val
        self.output_features = output_file
        self.output_features_val = output_file_val
        self.predict_seq = predict_seq_file
        self.predict_data = predict_file
        self.predict_output_features = []
        #self.args = args

        #print('xxx',self.args)
        train_dataset_list = []
        val_dataset_list = []
        test_dataset_list = []
        predict_dataset_list = []
        train_dataset = Dataset(self.sequence, self.input_features, self.output_features, seq_len, target_length)
        val_dataset = Dataset(self.sequence_val, self.input_features_val, self.output_features_val, seq_len,
                              target_length)
        test_dataset = Dataset(self.sequence_val, self.input_features_val,self.output_features_val, seq_len,
                               target_length)
        predict_dataset = Dataset(self.predict_seq, self.predict_data, self.predict_output_features, seq_len,
                                  target_length)
        # print(self.sequence[1:10],self.input_features, self.output_features)
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)
        test_dataset_list.append(test_dataset)
        predict_dataset_list.append(predict_dataset)

        # print('train_dataset_list',train_dataset_list)
        self.train_dataset = ConcatDataset(train_dataset_list)
        # print('ddd',self.train_dataset[10000])
        self.val_dataset = ConcatDataset(val_dataset_list)
        self.test_dataset = ConcatDataset(test_dataset_list)
        self.predict_dataset = ConcatDataset(predict_dataset_list)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=1
        )

        # print('train_loader',loader)
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=1
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=1
        )
        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self.predict_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=1
        )
        return loader


class TrainModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.get_model(args)
        self.args = args
        self.criterion = nn.MSELoss(reduction='sum')
        self.pcc = MeanPearsonCorrCoefPerChannel(1)

    def forward(self, x):
        pred = self.model(x)
        return pred

    def proc_batch(self, batch):
        seq = batch['sequence']
        epi = []
        #print('xxx',len(batch['input_features']))
        for i in range(0, args.feature_number):
            epi.append(batch['input_features'][i].unsqueeze(2))
        '''
        epi = batch['input_features'].unsqueeze(2)
        #print('batch',batch['input_features'].unsqueeze(2))
        epi_2 = batch['input_features_2'].unsqueeze(2)
        '''

        targets = batch['output_features']
        inputs = seq
        for i in range(0, args.feature_number):
            inputs = torch.cat([inputs, epi[i]], dim=2)

        targets = targets.float()
        '''
        print('input_shape',inputs.shape)
        print('inputs',inputs)
        print('target_shape',targets.shape)
        print('targets',targets)
        '''
        return inputs, targets

    def pro_batch_predict(self, batch):
        seq = batch['sequence']
        # epi = batch['input_features'].unsqueeze(2)
        # epi_2 = batch['input_features_2'].unsqueeze(2)
        # inputs = torch.cat([seq, epi], dim=2)
        # inputs = torch.cat([inputs, epi_2], dim=2)

        inputs = seq
        epi = []
        for i in range(0, args.feature_number):
            epi.append(batch['input_features'][i].unsqueeze(2))
        for i in range(0, args.feature_number):
            inputs = torch.cat([inputs, epi[i]], dim=2)
        return inputs

    def training_step(self, batch, batch_idx):
        count_seq = batch['count_N'].sum()
        inputs, targets = self.proc_batch(batch)
        input_2 = inputs.permute(2, 0, 1)
        #print('input2',input_2[0],input_2[0].shape)
        # pc = []
        # for index in range(0,25):
        #     pc.append(str(self.pcc(input_2[index], targets).mean()))
        #
        # # 使用正则表达式提取每个字符串中的浮点数部分
        # float_values = [s.split(',')[0].replace('tensor(','') for s in pc]
        #
        # # 将浮点数部分用制表符连接起来，并以换行符结尾
        # #pc_f.write("\t".join(float_values) + "\n")

        pred = self(inputs)
        # print(targets,pred)
        # print(targets.shape, pred.shape)
        #print('pred',pred,pred.shape)
        #print('targets',targets,targets.shape)
        loss = self.criterion(pred, targets)/count_seq
        # print(loss)
        # loss = loss.mean()
        # print(loss)
        pcc = self.pcc(pred, targets).mean()
        ranks1 = pred.argsort().argsort().type(torch.float32)
        ranks2 = targets.argsort().argsort().type(torch.float32)
        scc = self.pcc(ranks1, ranks2).mean()
        metrics = {
            'loss/train_step': loss,
            'pearson/train_step': pcc,
            'spearman/train_step': scc
        }
        self.log_dict(metrics, batch_size=inputs.shape[0], prog_bar=True, sync_dist=True)
        return {'loss': loss, 'pcc': pcc, 'scc': scc}

    def validation_step(self, batch, batch_idx):
        count_seq = batch['count_N'].sum()
        inputs, targets = self.proc_batch(batch)
        pred = self(inputs)
        # loss = self.criterion(pred, targets).mean()
        loss = self.criterion(pred, targets)/count_seq
        pcc = self.pcc(pred, targets).mean()
        ranks1 = pred.argsort().argsort().type(torch.float32)
        ranks2 = targets.argsort().argsort().type(torch.float32)
        scc = self.pcc(ranks1, ranks2).mean()
        return {'loss': loss, 'pcc': pcc, 'scc': scc}

    def test_step(self, batch, batch_idx):
        count_seq = batch['count_N'].sum()
        inputs, targets = self.proc_batch(batch)
        pred = self(inputs)
        # loss = self.criterion(pred, targets).mean()
        loss = self.criterion(pred, targets)/count_seq
        pcc = self.pcc(pred, targets).mean()

        ranks1 = pred.argsort().argsort().type(torch.float32)
        ranks2 = targets.argsort().argsort().type(torch.float32)
        scc = self.pcc(ranks1, ranks2).mean()
        return {'loss': loss, 'pcc': pcc, 'scc': scc}

    def predict_step(self, batch, batch_idx):
        inputs = self.pro_batch_predict(batch)
        # print('aa',inputs.shape)
        pred = self(inputs)

        return pred

    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        # log.write('train' + '\t' + str(ret_metrics['loss']) + '\t' + str(ret_metrics['pcc']) + '\t' + str(ret_metrics['scc']) + '\n')
        metrics = {'train_loss': ret_metrics['loss'], 'train_pcc': ret_metrics['pcc'], 'train_scc': ret_metrics['scc']
                   }
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        log.write('valid' + '\t' + str(ret_metrics['loss']) + '\t' + str(ret_metrics['pcc']) + '\t' + str(ret_metrics['scc']) + '\n')
        metrics = {'val_loss': ret_metrics['loss'], 'val_pcc': ret_metrics['pcc'], 'val_scc': ret_metrics['scc']
                   }
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

    def test_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        log.write('test' + '\t' + str(ret_metrics['loss']) + '\t' + str(ret_metrics['pcc']) + '\t' + str(
            ret_metrics['scc']) + '\n')
        metrics = {'test_loss': ret_metrics['loss'], 'test_pcc': ret_metrics['pcc'], 'test_scc': ret_metrics['scc']
                   }
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

    def _shared_epoch_end(self, step_outputs):
        pcc = torch.tensor([out['pcc'] for out in step_outputs])
        loss = torch.tensor([out['loss'] for out in step_outputs])
        scc = torch.tensor([out['scc'] for out in step_outputs])
        pcc = pcc[~torch.isnan(pcc)]
        scc = scc[~torch.isnan(scc)]
        avg_pcc = pcc.mean()
        avg_loss = loss.mean()
        avg_scc = scc.mean()
        return {'loss': avg_loss, 'pcc': avg_pcc, 'scc': avg_scc}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)  # final
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2000,  # self.hparams.warmup_steps
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'get_cosine_schedule_with_warmup',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}

    def get_model(self, args):
        model_name = args.model_type
        num_genomic_features = args.feature_number
        ModelClass = getattr(models, model_name)
        model = ModelClass(num_genomic_features, mid_hidden=10,seq_len=args.seq_length)
        return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Translatomer')

    parser.add_argument('--seed', dest='run_seed', default=2077,
                        type=int,
                        help='Random seed for training')

    parser.add_argument('--run_type', default='train',
                        type=str,
                        help='The pattern of the model')

    parser.add_argument('--predict_result_path', default='E:/Python项目文件夹/Translatomer/predict_result/predict_result.txt',
                        type=str,
                        help='The pattern of the model')

    parser.add_argument('--feature_number',
                        default= 23,
                        type=int,
                        help='The number of input feature')

    parser.add_argument('--sequence_data', default='C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输入数据_序列_100bp.txt',
                        type=str,
                        help='The sequence data path')

    parser.add_argument('--input_feature_data',
                        default='C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输入数据_特征_100bp.txt',
                        type=str,
                        help='The input feature path')

    parser.add_argument('--input_feature_2_data',
                        default='E:/Python项目文件夹/Translatomer/data/fin_input_feature_2_data_addjarvis_norm.txt',
                        type=str,
                        help='The second input feature path')

    parser.add_argument('--output_feature_data',
                        default='C:/Users/xiaxq/Desktop/冷热点预测课题/feature/输出数据_冷热点分数_100bp.txt',
                        type=str,
                        help='The output feature path')

    parser.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                        help='Path to the model checkpoint')

    parser.add_argument('--model-type', dest='model_type', default='TransModel',
                        help='Transformer')
    parser.add_argument('--fold', dest='n_fold', default='0',
                        help='Which fold of the model training')

    # Training Parameters
    parser.add_argument('--patience', dest='trainer_patience', default=8,
                        type=int,
                        help='Epoches before early stopping')
    parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=10,
                        type=int,
                        help='Max epochs')
    parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=5,
                        type=int,
                        help='Top n models to save')
    parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=1,
                        type=int,
                        help='Number of GPUs to use')
    # Dataloader Parameters
    parser.add_argument('--batch-size', dest='dataloader_batch_size', default=32,
                        type=int,
                        help='Batch size')
    parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                        action='store_false',
                        help='Using ddp, adjust batch size')
    parser.add_argument('--num-workers', dest='dataloader_num_workers', default=1,
                        type=int,
                        help='Dataloader workers')
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--seq_length', type=int, default=100)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    seq_len = args.seq_length
    target_length = args.seq_length

    sequence_data = []
    sequence_data_val = []
    input_data = []
    input_data_2 = []
    output_data = []
    input_data_val = []
    input_data_2_val = []
    output_data_val = []
    predict_seq_data = []
    predict_data = []
    predict_data_2 = []

    '''train'''
    if args.run_type == 'train':
        idx = []

        input_list = []
        input_list_val = []
        input_list_2 = []
        input_list_val_2 = []
        count = 0
        with open(args.input_feature_data, 'r') as f:
            for line in f:
                data = line.replace('\n', '').split(' ')
                if random.randint(0, 20) == 1:
                    input_list_val.append([])

                    for i in data:
                        i = (i.replace('[', '')).replace(']', '')
                        i = i.replace('nan', '0')
                        # input_list_val[-1].append(float(i))
                        input_list_val[-1].append(i)
                    idx.append(count)

                else:
                    input_list.append([])
                    for i in data:
                        i = (i.replace('[', '')).replace(']', '')
                        i = i.replace('nan', '0')
                        # input_list[-1].append(float(i))
                        input_list[-1].append(i)
                count += 1

        #my_tensor = torch.Tensor(input_list)
        #input_data.append(my_tensor)
        input_data.append(input_list)
        #print(input_data)
        # my_tensor_val = torch.Tensor(input_list_val)
        # input_data_val.append(my_tensor_val)
        input_data_val.append(input_list_val)
        f.close()

        # count = 0
        # with open(args.input_feature_2_data, 'r') as f:
        #     for line in f:
        #         data = line.replace('\n', '').split(' ')
        #
        #         if count in idx:
        #             input_list_val_2.append([])
        #             for i in data:
        #                 i = (i.replace('[','')).replace(']','')
        #                 input_list_val_2[-1].append(float(i))
        #         else:
        #             input_list_2.append([])
        #             for i in data:
        #                 i = (i.replace('[', '')).replace(']', '')
        #                 input_list_2[-1].append(float(i))
        #         count += 1
        #
        #
        # my_tensor = torch.Tensor(input_list_2)
        # input_data_2.append(my_tensor)
        # my_tensor_val = torch.Tensor(input_list_val_2)
        # input_data_2_val.append(my_tensor_val)
        # f.close()

        output_list = []
        output_list_val = []
        count = 0
        with open(args.output_feature_data, 'r') as out:
            for line in out:
                data = line.replace('\n', '').split(' ')
                if count in idx:
                    output_list_val.append([])
                    for i in data:
                        output_list_val[-1].append(float(i))
                else:
                    output_list.append([])
                    for i in data:
                        output_list[-1].append(float(i))
                count += 1

        my_tensor = torch.Tensor(output_list)
        output_data.append(my_tensor)
        my_tensor_val = torch.Tensor(output_list_val)
        output_data_val.append(my_tensor_val)
        out.close()

        with open(args.sequence_data, 'r') as seq:
            count = 0
            for line in seq:
                if count in idx:
                    sequence_data_val.append(line.replace('\n', ''))
                else:
                    sequence_data.append(line.replace('\n', ''))
                count += 1

        seq.close()

    if args.run_type == 'predict':
        '''predict'''
        with open(args.sequence_data, 'r') as pre_seq:
            for line in pre_seq:
                predict_seq_data.append(line.split('\t')[-1].replace('\n', ''))

            pre_seq.close()

        predict_data_list = []
        with open(args.input_feature_data, 'r') as pre_data:
            for line in pre_data:
                data = line.replace('\n', '').split(' ')
                predict_data_list.append([])
                for i in data:
                    i = (i.replace('[', '')).replace(']', '')
                    i = i.replace('nan', '0')
                    predict_data_list[-1].append(i)

        #my_tensor = torch.Tensor(predict_data_list)
        predict_data.append(predict_data_list)
        # print(len(predict_data[0][0]))
        for i in range(0,len(predict_data[0])):
            if len(predict_data[0][i]) != 100:
                print(len(predict_data[0][i]))
                print(i)




        pre_data.close()


        # predict_data_list_2 = []
        # with open(args.input_feature_2_data, 'r') as pre_data:
        #     for line in pre_data:
        #         data = line.replace('\n', '').split(' ')
        #         predict_data_list_2.append([])
        #         for i in data:
        #             predict_data_list_2[-1].append(float(i))
        #
        # my_tensor = torch.Tensor(predict_data_list_2)
        # predict_data_2.append(my_tensor)
        # pre_data.close()

    pl.seed_everything(args.run_seed, workers=True)
    dataset = DataModule(
        sequence=sequence_data,
        sequence_val=sequence_data_val,
        input_file=input_data,
        input_file_val=input_data_val,
        # input_file_2=input_data_2,
        # input_file_val_2=input_data_2_val,
        output_file=output_data,
        output_file_val=output_data_val,
        predict_seq_file=predict_seq_data,
        predict_file=predict_data,
        # predict_file_2=predict_data_2,
        seq_len=seq_len,
        target_length=target_length,
        batch_size=args.dataloader_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if args.run_type == 'train':
        # loading data
        if args.checkpoint:
            model = TrainModule.load_from_checkpoint(args.checkpoint)
            trainer = Trainer()
            trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())
            trainer.test(model, dataset.test_dataloader())

        else:
            # Early_stopping
            early_stop_callback = callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=0.00,
                                                          patience=args.trainer_patience,
                                                          verbose=False,
                                                          mode="min")
            # Checkpoints
            checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}/models',
                                                            save_top_k=args.trainer_save_top_n,
                                                            monitor='val_loss',
                                                            mode='min')
            # LR monitor
            lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
            csv_logger = pl.loggers.CSVLogger(save_dir=f'{args.run_save_path}/csv')
            all_loggers = csv_logger

            pl.seed_everything(args.run_seed, workers=True)
            pl_module = TrainModule(args)
            trainer = Trainer(gpus=1)
            trainer.fit(pl_module, dataset.train_dataloader(), dataset.test_dataloader())
            trainer.test(pl_module, dataset.test_dataloader())
    elif args.run_type == 'predict':
        model = TrainModule.load_from_checkpoint(args.checkpoint)
        trainer = Trainer()
        pred = trainer.predict(model, dataset.predict_dataloader())
        pre_pos = []
        predict_seq_data = []
        with open(args.predict_result_path,'w') as result_file,\
            open(args.sequence_data, 'r') as pre_seq:
            for line in pre_seq:
                pre_pos.append(line.split('\t')[0])
                predict_seq_data.append(line.split('\t')[-1].replace('\n', ''))
            count = 0
            for elem in pred:
                elem = elem.numpy()
                for re in elem:
                    for idx in range(0,len(re)):
                        header = [
                            pre_pos[count].split('-')[0],
                            str(int(pre_pos[count].split('-')[1]) + idx),
                            str(int(pre_pos[count].split('-')[1]) + idx),
                            predict_seq_data[count][idx],
                            str(re[idx])
                        ]
                        if predict_seq_data[count][idx] != 'N':
                            result_file.write(f"{'\t'.join(header)}\n")
                    count += 1
        result_file.close()
        pre_seq.close()




    else:
        print(f'Error! the model does not have mode \'{args.model_type}\'')

