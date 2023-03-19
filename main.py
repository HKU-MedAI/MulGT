#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function
import json
import os
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
# import GradNorm
from models.MulGT import MulGT
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
torch.cuda.current_device()
print(print(torch.cuda.is_available()))
import warnings
warnings.filterwarnings("ignore")
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from utils.dataset import GraphDataset
from torch.utils.tensorboard import SummaryWriter
from helper import Trainer, Evaluator, collate, preparefeatureLabel
from option import Options
from sklearn.model_selection import StratifiedKFold, train_test_split
import time as sys_time



def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def reuturn_data_label(data_index, patiens_list, label_list, wsi_list, batch_size):
    wsiidx = []
    label_array = []
    for item in data_index:
        temp_patient_name = patiens_list[item]
        for wsi in wsi_list:
            if temp_patient_name in wsi:
                label_array.append(label_list[item])
                wsiidx.append(wsi)

    return wsiidx, label_array


def reuturn_data_label_patient_wise(data_index, patiens_list, label_list, wsi_list, batch_size):
    label_array = []
    for item in data_index:
        label_array.append(label_list[item])
    return data_index, label_array


def write_dataset_txt(wsi_list, label_list, filename):
    with open(filename, "w") as f:
        for i in range(len(wsi_list)-1):
            sample = wsi_list[i]
            label = label_list[i]
            f.write(sample + '\t' + str(label)+'\n')
        sample = wsi_list[len(wsi_list)-1]
        label = label_list[len(wsi_list)-1]
        f.write(sample + '\t' + str(label))
    f.close()
    print("finish write: "+filename)


def return_acc(prediction_array, label_array):
    prediction_array = np.array(prediction_array)
    label_array = np.array(label_array)
    correct_num = (prediction_array == label_array).sum()
    len_array = len(prediction_array)
    return correct_num / len_array


def return_auc(possibility_array, label_array, n_class):
    np_label_array = np.zeros((len(label_array), args.n_class))
    for i in range(len(label_array)):
        np_label_array[i][label_array[i]] = 1
    possibility_array = np.array(possibility_array)

    aucs = []
    for c in range(0, n_class):
        label = np_label_array[:, c]
        prediction = possibility_array[:, c]
        # fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        # fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        # thresholds.append(threshold)
        # thresholds_optimal.append(threshold_optimal)

    total_auc_macro = roc_auc_score(np_label_array, possibility_array, average="macro")
    total_auc_micro = roc_auc_score(np_label_array, possibility_array, average="micro")
    return aucs, total_auc_macro, total_auc_micro


def val_test_block(model, dataloader_val, stage_data, args):
    model.eval()
    with torch.no_grad():
        total = 0.
        batch_idx = 0
        label_array_for_val_test = []
        prediction_array_for_val_test = []
        possibility_array_for_val_test = []
        label_array_stage_for_val_test = []
        prediction_array_stage_for_val_test = []
        possibility_array_stage_for_val_test = []
        total_loss_for_test = 0

        for i_batch, sample_batched in enumerate(dataloader_val):
            sample_stage = sample_batched['id'][0].split('-')[0] + '-' + sample_batched['id'][0].split('-')[1] + '-' + \
                           sample_batched['id'][0].split('-')[2]
            stage_label = stage_data[sample_stage]['cancer_stage']

            node_feat, labels, adjs, masks = preparefeatureLabel(sample_batched['image'], sample_batched['label'],
                                                                 sample_batched['adj_s'])

            prob, preds, labels, prob_stage, preds_stage, labels_stage, stage_loss, subtype_loss, reg_loss \
                = model.forward(node_feat, labels, adjs, masks, stage_label)
            loss = (subtype_loss + stage_loss + reg_loss) / 3.0
            loss = loss.mean()

            total += len(labels)

            total_loss_for_test += loss

            label_array_for_val_test.append(labels.cpu().item())
            prediction_array_for_val_test.append(preds.cpu().item())
            possibility_array_for_val_test.append(prob.squeeze().cpu().detach().numpy())

            label_array_stage_for_val_test.append(labels_stage.cpu().item())
            prediction_array_stage_for_val_test.append(preds_stage.cpu().item())
            possibility_array_stage_for_val_test.append(prob_stage.squeeze().cpu().detach().numpy())

        acc_for_test = return_acc(prediction_array_for_val_test, label_array_for_val_test)
        auc_for_test, macro_auc_for_test, micro_auc_for_test = return_auc(
            possibility_array_for_val_test, label_array_for_val_test, args.n_class)

        f1_for_test = f1_score(label_array_for_val_test, prediction_array_for_val_test, average='weighted')

        acc_stage_for_test = return_acc(prediction_array_stage_for_val_test, label_array_stage_for_val_test)
        auc_stage_for_test, macro_auc_stage_for_test, micro_auc_stage_for_test = return_auc(
            possibility_array_stage_for_val_test, label_array_stage_for_val_test, args.stage_class)

        f1_stage_for_test = f1_score(label_array_stage_for_val_test, prediction_array_stage_for_val_test, average='weighted')

        return acc_for_test, auc_for_test, macro_auc_for_test, micro_auc_for_test, \
               acc_stage_for_test, auc_stage_for_test, macro_auc_stage_for_test, micro_auc_stage_for_test, \
               total_loss_for_test, f1_for_test, f1_stage_for_test


class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(sys_time.strftime('%Y-%m-%d-%H-%M-%S'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


args = Options().parse()

n_class = args.n_class

torch.cuda.synchronize()

data_path = args.data_path

model_path = args.model_path
if not os.path.isdir(model_path):
    os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path):
    os.mkdir(log_path)
task_name = args.task_name

if args.log:
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)
    writer = SummaryWriter(comment='_'+task_name)

print(task_name)
###################################
train = args.train
test = args.test
graphcam = args.graphcam
print("train:", train, "test:", test, "graphcam:", graphcam)
print("\n")

patient_and_label_path = args.patient_and_label_path
patient_and_label = json.loads(pd.read_csv(patient_and_label_path).to_json(orient="index"))
wsi_and_label_path = args.wsi_and_label_path
wsi_and_label = json.loads(pd.read_csv(wsi_and_label_path).to_json(orient="index"))

patient_list = []
label_list = []
for item in patient_and_label:
    patient_list.append(patient_and_label[item]['patient_id'])
    label_list.append(patient_and_label[item]['cancer_classification'])

wsi_list = []
wis_label_list = []
for item in wsi_and_label:
    wsi_list.append(wsi_and_label[item]['wsi_id'])
    wis_label_list.append(wsi_and_label[item]['cancer_classification'])

stage_data = dict()
for item in patient_and_label:
    stage_record = dict()
    stage_record['cancer_stage'] = patient_and_label[item]['cancer_stage']
    stage_data[patient_and_label[item]['patient_id']] = stage_record

repeat_num = args.repeat_num

all_fold_test_auc_subtype = []
all_fold_test_acc_subtype = []
all_fold_test_f1_subtype = []
all_fold_test_auc_stage = []
all_fold_test_acc_stage = []
all_fold_test_f1_stage = []
all_fold_val_auc_subtype = []
all_fold_val_acc_subtype = []
all_fold_val_f1_subtype = []
all_fold_val_auc_stage = []
all_fold_val_acc_stage = []
all_fold_val_f1_stage = []

since = time.time()
for repeat_num_temp in range(repeat_num):
    pre_result = {}
    fold_auc_test_subtype = []
    fold_acc_test_subtype = []
    fold_f1_test_subtype = []
    fold_auc_test_stage = []
    fold_acc_test_stage = []
    fold_f1_test_stage = []
    fold_auc_val_subtype = []
    fold_acc_val_subtype = []
    fold_f1_val_subtype = []
    fold_auc_val_stage = []
    fold_acc_val_stage = []
    fold_f1_val_stage = []

    batch_size = args.batch_size

    seed = 0
    setup_seed(repeat_num_temp)
    kf = StratifiedKFold(n_splits=args.fold_num, shuffle=True, random_state=seed)
    fold_num = 0
    print('\nseed', seed, 'repeat_num_temp', repeat_num_temp)

    for train_index, test_index in kf.split(patient_list, label_list):
        fold_num = fold_num + 1
        print('fold:', fold_num)
        best_auc_subtype_val_fold = 0.0
        best_auc_stage_val_fold = 0.0
        best_auc_subtype_test_fold = 0.0
        best_auc_stage_test_fold = 0.0

        print("preparing datasets and dataloaders......")
        train_index, train_label = reuturn_data_label_patient_wise(data_index=list(train_index),
                                                                   patiens_list=patient_list, label_list=label_list,
                                                                   wsi_list=wsi_list, batch_size=1)
        train_index, val_index, _, _ = train_test_split(train_index, train_label, test_size=0.25,
                                                                    random_state=seed, stratify=train_label)
        data_for_train, label_for_train = reuturn_data_label(data_index=train_index,
                                                             patiens_list=patient_list, label_list=label_list,
                                                             wsi_list=wsi_list, batch_size=1)
        write_dataset_txt(data_for_train, label_for_train, "./train_set.txt")

        data_for_val, label_for_val = reuturn_data_label(data_index=val_index,
                                                         patiens_list=patient_list, label_list=label_list,
                                                         wsi_list=wsi_list, batch_size=1)
        write_dataset_txt(data_for_val, label_for_val, "./val_set.txt")

        data_for_test, label_for_test = reuturn_data_label(data_index=list(test_index),
                                                           patiens_list=patient_list, label_list=label_list,
                                                           wsi_list=wsi_list, batch_size=1)
        write_dataset_txt(data_for_test, label_for_test, "./test_set.txt")

        test_patient_list = list()
        for patient in data_for_test:
            test_patient_list.append(patient)
        train_sample_num = len(data_for_train)
        val_sample_num = len(data_for_val)
        test_sample_num = len(data_for_test)
        print("train_WSI_num: " + str(train_sample_num)
              + " val_WSI_num: " + str(val_sample_num)
              + " test_WSI_num: " + str(test_sample_num))
        print("train_patient_num: " + str(len(train_index))
              + " val_patient_num: " + str(len(val_index))
              + " test_patient_num: " + str(len(test_index)))



        ids_train = open(args.train_set).readlines()
        dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train, feature_extractor=args.feature_extractor,
                                         train_noise=args.train_noise, survival_data=stage_data)
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, num_workers=8,
                                                           collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
        total_train_num = len(dataloader_train) * batch_size
        ids_val = open(args.val_set).readlines()
        dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val, feature_extractor=args.feature_extractor,
                                   survival_data=stage_data)
        dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, num_workers=8,
                                                     collate_fn=collate, shuffle=False, pin_memory=True)
        total_val_num = len(dataloader_val) * batch_size
        ids_test = open(args.test_set).readlines()
        dataset_test = GraphDataset(os.path.join(data_path, ""), ids_test, feature_extractor=args.feature_extractor,
                                   survival_data=stage_data)
        dataloader_test= torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, num_workers=8,
                                                     collate_fn=collate, shuffle=False, pin_memory=True)
        total_test_num = len(dataloader_test) * batch_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ##### creating models #############
        print("creating models......\n")

        num_epochs = args.num_epochs
        learning_rate = args.lr

        model = MulGT(subtype_class=args.n_class, stage_class=args.stage_class,
                      input_dim=args.input_dim,
                      mlp_head=args.mlp_head, args=args)

        if args.resume:
            print('load model{}'.format(args.resume))
            model.load_state_dict(torch.load(args.resume))

        if torch.cuda.is_available():
            model = model.cuda()

        if args.baseline:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)  # best:5e-4, 4e-3
        else:
            stage_para = []
            rest_para = []
            for name, param in model.named_parameters():
                if param.requires_grad and (
                        'survival' in name or 'sending' in name or 'pos' in name or 'token' in name):
                    stage_para.append(param)
                    # print(name)
                else:
                    rest_para.append(param)
            optimizer = torch.optim.Adam([
                {'params': rest_para, 'lr': learning_rate, 'weight_decay': 5e-4},
                {'params': stage_para, 'lr': learning_rate * 0.1, 'weight_decay': 5e-4}
            ])

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100],
                                                         gamma=0.1)  # gamma=0.3  # 30,90,130 # 20,90,130 -> 150

        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.
            total = 0.

            prediction_array_subtype = []
            label_array_subtype = []
            possibility_array_subtype = []
            prediction_array_stage = []
            label_array_stage = []
            possibility_array_stage = []
            total_loss_for_train = 0

            current_lr = optimizer.param_groups[0]['lr']

            if train:
                batch_loss = 0
                for i_batch, sample_batched in enumerate(dataloader_train):
                    scheduler.step(epoch)

                    sample_stage = sample_batched['id'][0].split('-')[0] + '-' + sample_batched['id'][0].split('-')[1] + '-' + \
                                    sample_batched['id'][0].split('-')[2]
                    stage_label = stage_data[sample_stage]['cancer_stage']

                    node_feat, labels, adjs, masks = preparefeatureLabel(sample_batched['image'], sample_batched['label'],
                                                                         sample_batched['adj_s'])
                    prob, preds, labels, prob_stage, preds_stage, labels_stage, stage_loss, subtype_loss, reg_loss \
                        = model.forward(node_feat, labels, adjs, masks, stage_label)

                    if args.task_type == "multi":
                        if args.grad_norm:
                            stage_loss = stage_loss.unsqueeze(0)
                            subtype_loss = subtype_loss.unsqueeze(0)
                            loss = torch.cat([stage_loss, subtype_loss],0)
                            loss = grad_norm(loss)/2.0
                            stage_loss = stage_loss.squeeze(0)
                            subtype_loss = subtype_loss.squeeze(0)
                        else:
                            loss = (stage_loss + subtype_loss)/2.0
                    elif args.task_type == "subtype":
                        loss = subtype_loss
                    elif args.task_type == "stage":
                        loss = stage_loss

                    if args.pool_method == "dense_diff_pool" or args.pool_method == "dense_mincut_pool" or args.reg_loss:
                        loss = loss + reg_loss

                    loss = loss.mean()
                    batch_loss += loss

                    if i_batch % batch_size == 0 or i_batch == len(dataloader_train) - 1:
                        batch_loss = batch_loss/batch_size
                        optimizer.zero_grad()

                        if args.grad_norm:
                            batch_loss.backward(retain_graph=True)
                            grad_norm.additional_forward_and_backward(grad_norm_weights=model.W_O, total_loss=batch_loss)
                            if i_batch > 256:
                                break
                        else:
                            batch_loss.backward()

                        optimizer.step()
                        batch_loss = 0

                        torch.cuda.empty_cache()

                    train_loss += loss
                    total += len(labels)

                    total_loss_for_train += loss

                    label_array_subtype.append(labels.cpu().item())
                    prediction_array_subtype.append(preds.cpu().item())
                    possibility_array_subtype.append(prob.squeeze().cpu().detach().numpy())

                    label_array_stage.append(labels_stage.cpu().item())
                    prediction_array_stage.append(preds_stage.cpu().item())
                    possibility_array_stage.append(prob_stage.squeeze().cpu().detach().numpy())

                acc = return_acc(prediction_array_subtype, label_array_subtype)
                auc, macro_auc, micro_auc = return_auc(possibility_array_subtype, label_array_subtype, args.n_class)
                f1 = f1_score(label_array_subtype, prediction_array_subtype, average='weighted')

                acc_stage = return_acc(prediction_array_stage, label_array_stage)
                auc_stage, macro_auc_stage, micro_auc_stage = return_auc(possibility_array_stage, label_array_stage, args.stage_class)
                f1_stage = f1_score(label_array_subtype, prediction_array_subtype, average='weighted')

            acc_for_test, auc_for_test, macro_auc_for_test, micro_auc_for_test, \
            acc_stage_for_test, auc_stage_for_test, macro_auc_stage_for_test, micro_auc_stage_for_test, \
            total_loss_for_test, f1_for_test, f1_stage_for_test = val_test_block(model, dataloader_test, stage_data, args)

            acc_for_val, auc_for_val, macro_auc_for_val, micro_auc_for_val, \
            acc_stage_for_val, auc_stage_for_val, macro_auc_stage_for_val, micro_auc_stage_for_val, \
            total_loss_for_val, f1_for_val, f1_stage_for_val = val_test_block(model, dataloader_val, stage_data, args)

            print("epoch：{:2d}:\n".format(epoch))
            print("train_loss：{:.4f},  val_loss：{:.4f},  test_loss：{:.4f}".format(
                    total_loss_for_train/train_sample_num, total_loss_for_val/val_sample_num,
                    total_loss_for_test/test_sample_num,))
            print("subtype_loss：{:.4f},  stage_loss：{:.4f},  reg_loss：{:.4f}".format(subtype_loss, stage_loss, reg_loss))

            print("train_subtype_acc：\t{:.4f}\t train_subtype_auc：\t{:.4f}\t train_subtype_f1：\t{:.4f}\t"
                  "train_stage_acc：\t{:.4f}\t train_stage_auc：\t{:.4f}\t  train_stage_f1：\t{:.4f}".format(
                    acc, micro_auc, f1, acc_stage, micro_auc_stage, f1_stage))
            print("val_subtype_acc：\t{:.4f}\t val_subtype_auc：\t\t{:.4f}\t val_subtype_f1：\t{:.4f}\t"
                  "val_stage_acc：\t\t{:.4f}\t val_stage_auc：\t{:.4f}\t  val_stage_f1：\t{:.4f}".format(
                    acc_for_val, micro_auc_for_val, f1_for_val, acc_stage_for_val, micro_auc_stage_for_val, f1_stage_for_val))
            print("test_subtype_acc：\t{:.4f}\t test_subtype_auc：\t\t{:.4f}\t test_subtype_f1：\t{:.4f}\t"
                  "test_stage_acc：\t{:.4f}\t test_stage_auc：\t{:.4f}\t  test_stage_f1：\t{:.4f}".format(
                    acc_for_test, micro_auc_for_test, f1_for_test, acc_stage_for_test, micro_auc_stage_for_test, f1_stage_for_test))

            print("train subtype_auc: " + str(auc) + '\t' + "train stage_auc: " + str(auc_stage))
            print("val subtype_auc:   " + str(auc_for_val) + '\t' +"val stage_auc:   " + str(auc_stage_for_val))
            print("test subtype_auc:  " + str(auc_for_test) + '\t' +"test stage_auc:  " + str(auc_stage_for_test))

            if args.log:
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_training_loss',
                                  total_loss_for_train / train_sample_num, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num)+'/_training_subtype_acc',
                                  acc, global_step=epoch)
                writer.add_scalar(
                    'repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_training_subtype_auc',
                    micro_auc, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_training_stage_acc',
                                  acc_stage, global_step=epoch)
                writer.add_scalar(
                    'repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_training_stage_auc',
                    micro_auc_stage, global_step=epoch)

                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_test_loss',
                                  total_loss_for_test / test_sample_num, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_test_subtype_acc',
                                  acc_for_test, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_test_subtype_auc',
                                  micro_auc_for_test, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_test_stage_acc',
                                  acc_stage_for_test, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_test_stage_auc',
                                  micro_auc_stage_for_test, global_step=epoch)

                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_val_loss',
                                  total_loss_for_val / val_sample_num, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_val_subtype_acc',
                                  acc_for_val, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_val_subtype_auc',
                                  micro_auc_for_val, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_val_stage_acc',
                                  acc_stage_for_val, global_step=epoch)
                writer.add_scalar('repeat_' + str(repeat_num_temp) + '/_fold_' + str(fold_num) + '/_val_stage_auc',
                                  micro_auc_stage_for_val, global_step=epoch)

            time_elapsed = time.time() - since
            print('Time completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


            if micro_auc_for_val >= best_auc_subtype_val_fold and (micro_auc > 0.8 or epoch > (num_epochs - 10)):
                best_auc_subtype_val_fold = micro_auc_for_val
                best_acc_subtype_val_fold = acc_for_val
                best_auc_subtype_test_fold = micro_auc_for_test
                best_acc_subtype_test_fold = acc_for_test
                best_f1_subtype_test_fold = f1_for_test
                best_f1_subtype_val_fold = f1_for_val

                print("best_acc_plus_auc_test_repeat_"+str(repeat_num_temp)+"_fold_" + str(fold_num)
                      + ", save subtype_model")
                torch.save(model.state_dict(),
                           model_path + task_name + '_patient_wise_repeat_'+str(repeat_num_temp)+"_fold_"
                           + str(fold_num) + '_best_subtype.pth')

            if micro_auc_stage_for_val >= best_auc_stage_val_fold and (micro_auc_stage > 0.7 or epoch > (num_epochs - 10)):
                best_auc_stage_val_fold = micro_auc_stage_for_val
                best_acc_stage_val_fold = acc_stage_for_val
                best_auc_stage_test_fold = micro_auc_stage_for_test
                best_acc_stage_test_fold = acc_stage_for_test
                best_f1_stage_test_fold = f1_stage_for_test
                best_f1_stage_val_fold = f1_stage_for_val

                print("best_acc_plus_auc_test_repeat_"+str(repeat_num_temp)+"_fold_" + str(fold_num)
                      + ", save stage_model")
                torch.save(model.state_dict(),
                           model_path + task_name + '_patient_wise_repeat_'+str(repeat_num_temp)+"_fold_"
                           + str(fold_num) + '_best_stage.pth')

            log = "======================================================================================================================\n"
            print(log)

        fold_auc_test_subtype.append(best_auc_subtype_test_fold)
        fold_acc_test_subtype.append(best_acc_subtype_test_fold)
        fold_f1_test_subtype.append(best_f1_subtype_test_fold)
        fold_auc_test_stage.append(best_auc_stage_test_fold)
        fold_acc_test_stage.append(best_acc_stage_test_fold)
        fold_f1_test_stage.append(best_f1_stage_test_fold)
        fold_auc_val_subtype.append(best_auc_subtype_val_fold)
        fold_acc_val_subtype.append(best_acc_subtype_val_fold)
        fold_f1_val_subtype.append(best_f1_subtype_val_fold)
        fold_auc_val_stage.append(best_auc_stage_val_fold)
        fold_acc_val_stage.append(best_acc_stage_val_fold)
        fold_f1_val_stage.append(best_f1_stage_val_fold)

        print('subtype micro_auc: ', best_auc_subtype_test_fold)
        print('subtype acc of test: ', best_acc_subtype_test_fold)
        print('subtype f1 of test: ', best_f1_subtype_test_fold)
        print('stage micro_auc: ', best_auc_stage_test_fold)
        print('stage acc of test: ', best_acc_stage_test_fold)
        print('stage f1 of test: ', best_f1_stage_test_fold)

        print("\n")

    all_fold_test_auc_subtype.append(fold_auc_test_subtype)
    all_fold_test_acc_subtype.append(fold_acc_test_subtype)
    all_fold_test_f1_subtype.append(fold_f1_test_subtype)
    all_fold_test_auc_stage.append(fold_auc_test_stage)
    all_fold_test_acc_stage.append(fold_acc_test_stage)
    all_fold_test_f1_stage.append(fold_f1_test_stage)

    all_fold_val_auc_subtype.append(fold_auc_val_subtype)
    all_fold_val_acc_subtype.append(fold_acc_val_subtype)
    all_fold_val_f1_subtype.append(fold_f1_val_subtype)
    all_fold_val_auc_stage.append(fold_auc_val_stage)
    all_fold_val_acc_stage.append(fold_acc_val_stage)
    all_fold_val_f1_stage.append(fold_f1_val_stage)

    print('seed', seed)
    print('fold subtype auc:', fold_auc_test_subtype, ',mean:', np.mean(fold_auc_test_subtype))
    print('fold subtype acc:', fold_acc_test_subtype, ',mean:', np.mean(fold_acc_test_subtype))
    print('fold subtype f1:', fold_f1_test_subtype, ',mean:', np.mean(fold_f1_test_subtype))
    print('fold stage auc:', fold_auc_test_stage, ',mean:', np.mean(fold_auc_test_stage))
    print('fold stage acc:', fold_acc_test_stage, ',mean:', np.mean(fold_acc_test_stage))
    print('fold stage f1:', fold_f1_test_stage, ',mean:', np.mean(fold_f1_test_stage))


print('\nall subtype auc:')
for r in all_fold_test_auc_subtype:
    print(r)
print('mean subtype auc', np.mean(np.array(all_fold_test_auc_subtype)))
print('std subtype auc', np.std(np.array(all_fold_test_auc_subtype)))
print('\nall subtype acc:')
for r in all_fold_test_acc_subtype:
    print(r)
print('mean subtype acc', np.mean(np.array(all_fold_test_acc_subtype)))
print('std subtype acc', np.std(np.array(all_fold_test_acc_subtype)))
print('\nall subtype f1:')
for r in all_fold_test_f1_subtype:
    print(r)
print('mean subtype f1', np.mean(np.array(all_fold_test_f1_subtype)))
print('std subtype f1', np.std(np.array(all_fold_test_f1_subtype)))

print('\nall stage auc:')
for r in all_fold_test_auc_stage:
    print(r)
print('mean stage auc', np.mean(np.array(all_fold_test_auc_stage)))
print('std stage auc', np.std(np.array(all_fold_test_auc_stage)))
print('\nall stage acc:')
for r in all_fold_test_acc_stage:
    print(r)
print('mean stage acc', np.mean(np.array(all_fold_test_acc_stage)))
print('std stage acc', np.std(np.array(all_fold_test_acc_stage)))
print('\nall stage f1:')
for r in all_fold_test_f1_stage:
    print(r)
print('mean stage f1', np.mean(np.array(all_fold_test_f1_stage)))
print('std stage f1', np.std(np.array(all_fold_test_f1_stage)))

log = "======================================================================================================================\n"
print(log)
print('\nall val subtype auc:')
for r in all_fold_val_auc_subtype:
    print(r)
print('mean val subtype auc', np.mean(np.array(all_fold_val_auc_subtype)))
print('std val subtype auc', np.std(np.array(all_fold_val_auc_subtype)))
print('\nall val subtype acc:')
for r in all_fold_val_acc_subtype:
    print(r)
print('mean val subtype acc', np.mean(np.array(all_fold_val_acc_subtype)))
print('std val subtype acc', np.std(np.array(all_fold_val_acc_subtype)))
print('\nall val subtype f1:')
for r in all_fold_val_f1_subtype:
    print(r)
print('mean val subtype f1', np.mean(np.array(all_fold_val_f1_subtype)))
print('std val subtype f1', np.std(np.array(all_fold_val_f1_subtype)))

print('\nall val stage auc:')
for r in all_fold_val_auc_stage:
    print(r)
print('mean val stage auc', np.mean(np.array(all_fold_val_auc_stage)))
print('std val stage auc', np.std(np.array(all_fold_val_auc_stage)))
print('\nall val stage acc:')
for r in all_fold_val_acc_stage:
    print(r)
print('mean val stage acc', np.mean(np.array(all_fold_val_acc_stage)))
print('std val stage acc', np.std(np.array(all_fold_val_acc_stage)))
print('\nall val stage acc:')
for r in all_fold_val_f1_stage:
    print(r)
print('mean val stage f1', np.mean(np.array(all_fold_val_f1_stage)))
print('std val stage f1', np.std(np.array(all_fold_val_f1_stage)))


print('\n'+task_name)