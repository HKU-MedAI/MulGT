###########################################################################
# Created by: YI ZHENG
# Email: yizheng@bu.edu
# Copyright (c) 2020
###########################################################################

import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=2, help='classification classes')
        parser.add_argument('--survival_class', type=int, default=4, help='survival classes')
        parser.add_argument('--stage_class', type=int, default=2, help='stage classes')
        parser.add_argument('--data_path', type=str, default="/home/r10user3/Documents/GraphCAM/graphs",
                            help='path to dataset where images store')
        parser.add_argument('--survival_data_path', type=str,
                            default="/home/r10user3/Documents/GraphCAM/patient_survival_data.pkl",
                            help='path to dataset where images store')
        parser.add_argument("--patient_and_label_path", type=str,
                            default="/home/r10user3/Documents/GraphCAM/TCGA_clincical/KIDNEY_patient_and_label.csv",
                            help="Path of patient and label data")
        parser.add_argument("--wsi_and_label_path", type=str,
                            default="/home/r10user3/Documents/GraphCAM/TCGA_clincical/KIDNEY_wsi_and_label.csv",
                            help="Path of wsi and label data")
        parser.add_argument('--task_name', type=str, default='test', help='task name for naming saved model files and log files')
        parser.add_argument('--feature_extractor', type=str, default="KIDNEY_Kimia_20x_512x512", help='path for model')
        parser.add_argument('--task_type', type=str, default="multi", help='path for model', choices=["multi",
                                                                                                      "survival",
                                                                                                      "subtype",
                                                                                                      "stage"])
        parser.add_argument("--multi_task_method", type=str, default="Simple_Multi", help="training noise",
                            choices=["Cross_Attention", "Simple_Multi", "Exchange_Sending_Token",
                                     "Exchange_CLS_Token", "Exchange_Label"])
        parser.add_argument("--subtype_loss_ratio", type=float, default=1., help="subtype_loss_ratio")
        parser.add_argument("--survival_loss_ratio", type=float, default=1., help="subtype_loss_ratio")
        parser.add_argument("--stage_loss_ratio", type=float, default=1., help="subtype_loss_ratio")
        parser.add_argument("--task_pool", type=bool, default=False, help="task_pool")
        parser.add_argument("--pool_method", type=str, default="SAGPooling", help="task_pool")


        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of model training")
        parser.add_argument("--num_epochs", type=int, default=40, help="Cycle times of model training")
        parser.add_argument('--train_set', type=str, default="train_set.txt", help='train')
        parser.add_argument('--test_set', type=str, default="test_set.txt", help='test')
        parser.add_argument('--val_set', type=str, default="val_set.txt", help='validation')
        parser.add_argument('--model_path', type=str,
                            default="./graph_transformer/saved_models/",
                            help='path to trained model')
        parser.add_argument('--log_path', type=str,
                            default="./graph_transformer/runs/",
                            help='path to log files')
        parser.add_argument('--train', action='store_true', default=False, help='train only')
        parser.add_argument('--test', action='store_true', default=False, help='test only')
        parser.add_argument('--batch_size', type=int, default=1,
                            help='batch size for origin global image (without downsampling)')
        parser.add_argument('--log_interval_local', type=int, default=10, help='classification classes')
        parser.add_argument('--resume', type=str, default="", help='path for model')
        parser.add_argument('--graphcam', action='store_true', default=False, help='GraphCAM')
        parser.add_argument("--repeat_num", type=int, default=3, help="Number of repetitions of the experiment")
        parser.add_argument("--fold_num", type=int, default=5, help="fold number of this experiment")


        parser.add_argument("--train_noise", type=bool, default=False, help="training noise")
        parser.add_argument("--input_dim", type=int, default=1024, help="node embedding input dimension")

        parser.add_argument("--mlp_head", type=bool, default=False, help="mlp head")

        parser.add_argument("--log", type=bool, default=False, help="log")

        parser.add_argument("--task_proto", type=bool, default=False, help="log")
        parser.add_argument("--task_proto_first", type=bool, default=False, help="log")
        parser.add_argument("--baseline", type=bool, default=False, help="log")

        parser.add_argument("--phase1_node_num", type=int, default=150, help="node num after phase 1 pooling")
        parser.add_argument("--phase2_node_num", type=int, default=100, help="node num after phase 1 pooling")
        parser.add_argument("--drop_out", type=float, default=0.0, help="drop out rate of the model")

        parser.add_argument("--multisetKG", type=bool, default=False, help="multisetKG")

        parser.add_argument("--embed_dim", type=int, default=64, help="node embedding input dimension")
        parser.add_argument("--GCN_depth", type=int, default=1, help="GCN_depth")
        parser.add_argument("--subtypeT_depth", type=int, default=3, help="subtype Transformer depth")
        parser.add_argument("--stageT_depth", type=int, default=3, help="stage Transformer depth")

        parser.add_argument("--gnn_method", type=str, default="GCN", help="task_pool")
        parser.add_argument("--grad_norm", type=bool, default=False, help="task_pool")

        parser.add_argument("--linear_task_proto", type=bool, default=False, help="task_pool")
        parser.add_argument("--seperate_pooling", type=bool, default=False, help="task_pool")
        parser.add_argument("--GCN_MinCut", type=bool, default=False, help="task_pool")
        parser.add_argument("--reg_loss", type=bool, default=False, help="task_pool")

        parser.add_argument("--ViG_baseline_stage_test", type=bool, default=False, help="task_pool")
        parser.add_argument("--ViG_baseline_model", type=str, default=False, help="task_pool")
        parser.add_argument("--ViG_baseline_pool", type=str, default='SAGPooling', help="task_pool")

        parser.add_argument("--shared_proto", type=bool, default=False, help="task_pool")
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr

        # args.num_epochs = 120
        # args.lr = 1e-3

        if args.test:
            args.num_epochs = 1
        return args
