import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from .ViT import *
from .gcn import GCNBlock
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool, GraphMultisetTransformer, SAGPooling, \
    TopKPooling, dense_diff_pool, GraphConv, GATConv, GATv2Conv, GINConv, GINEConv
from torch.nn import Linear


class MulGT(nn.Module):
    def __init__(self, subtype_class, stage_class, input_dim=512,
                 mlp_head=False, args=None):
        super(MulGT, self).__init__()

        self.embed_dim = args.embed_dim
        self.num_layers = 3
        self.phase1_node_num = args.phase1_node_num
        self.phase2_node_num = args.phase2_node_num

        self.subtype_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.stage_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.subtype_transformer = VisionTransformer(num_classes=subtype_class, embed_dim=self.embed_dim,
                                                      mlp_head=mlp_head, depth=args.subtypeT_depth,
                                                      drop_rate=args.drop_out, attn_drop_rate=args.drop_out)
        self.stage_transformer = VisionTransformer(num_classes=stage_class, embed_dim=self.embed_dim,
                                                      mlp_head=mlp_head, depth=args.stageT_depth,
                                                      drop_rate=args.drop_out, attn_drop_rate=args.drop_out)

        self.criterion = nn.CrossEntropyLoss()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1

        self.gnn_method = args.gnn_method
        self.GCN_depth = args.GCN_depth

        self.conv1 = GCNBlock(input_dim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding,
                              dropout=args.drop_out, relu=0.)
        if self.GCN_depth > 1:
            self.gcns = nn.ModuleList()
            for i in range(self.GCN_depth-1):
                self.gcns.append(GCNBlock(self.embed_dim, self.embed_dim, self.bn, self.add_self,
                                          self.normalize_embedding, dropout=args.drop_out, relu=0.))
        else:
            self.GCN_depth = 1

        self.gcn_mincut = GCNBlock(self.embed_dim, self.embed_dim, self.bn, self.add_self,
                                   self.normalize_embedding, dropout=args.drop_out, relu=0.)
        self.separate_stage_pool = Linear(self.embed_dim, self.phase2_node_num)

        self.subtype_proto = nn.Parameter(torch.zeros(1, self.phase1_node_num, self.embed_dim))
        self.stage_proto = nn.Parameter(torch.zeros(1, self.phase1_node_num, self.embed_dim))
        self.subtype_qkv = Linear(self.embed_dim, self.embed_dim*2)
        self.stage_qkv = Linear(self.embed_dim, self.embed_dim*2)
        self.subtype_att = Attention(dim=self.embed_dim, num_heads=8, attn_drop=args.drop_out, proj_drop=args.drop_out)
        self.stage_att = Attention(dim=self.embed_dim, num_heads=8, attn_drop=args.drop_out, proj_drop=args.drop_out)
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

        self.task_norm_1 = LayerNorm(self.embed_dim, eps=1e-6)
        self.task_norm_2 = LayerNorm(self.embed_dim, eps=1e-6)
        self.task_norm_3 = LayerNorm(self.embed_dim, eps=1e-6)
        self.task_norm_4 = LayerNorm(self.embed_dim, eps=1e-6)
        self.subtype_att_W = Linear(self.embed_dim, self.embed_dim)
        self.stage_att_W = Linear(self.embed_dim, self.embed_dim)

        torch.nn.init.xavier_uniform_(self.subtype_proto)
        torch.nn.init.xavier_uniform_(self.stage_proto)


    def forward(self, node_feat, subtype_labels, adj, mask, stage_labels):
        X = node_feat

        X = mask.unsqueeze(2) * X

        X = self.conv1(X, adj, mask)

        X_subtype1, X_subtype2 = self.clone1(X, 2)
        X_subtype_qkv = self.subtype_qkv(self.subtype_proto)
        X_subtype_k, X_subtype_v = rearrange(X_subtype_qkv, 'b n (qkv h d) -> qkv b h n d', qkv=2, h=8)
        X_subtype = self.add1([X_subtype1, self.subtype_att_W(
            self.subtype_att(x=X_subtype2, out_k=X_subtype_k, out_v=X_subtype_v))])
        X_subtype = self.task_norm_2(X_subtype)
        X_subtype = self.task_norm_1(X_subtype + F.relu(X_subtype))

        X_stage1, X_stage2 = self.clone2(X, 2)
        X_stage_qkv = self.stage_qkv(self.stage_proto)
        X_stage_k, X_stage_v = rearrange(X_stage_qkv, 'b n (qkv h d) -> qkv b h n d', qkv=2, h=8)
        X_stage = self.add2([X_stage1, self.stage_att_W(
            self.stage_att(x=X_stage2, out_k=X_stage_k, out_v=X_stage_v))])
        X_stage = self.task_norm_4(X_stage)
        X_stage = self.task_norm_3(X_stage + F.relu(X_stage))

        idx_list = random.sample(range(0, X_subtype.shape[1] - 1), k=self.phase2_node_num)
        X_subtype = torch.index_select(X_subtype, 1, torch.tensor(idx_list).cuda())

        X_mincut = self.gcn_mincut(X_stage, adj, mask)
        s = self.separate_stage_pool(X_mincut)

        X_stage, adj, mc1, o1 = dense_mincut_pool(X_stage, adj, s, mask)

        b, _, _ = X_subtype.shape

        subtype_cls_token = self.subtype_cls_token.repeat(b, 1, 1)
        X_subtype = torch.cat([subtype_cls_token, X_subtype], dim=1)
        subtype_out, subtype_token = self.subtype_transformer(X_subtype)

        stage_cls_token = self.stage_cls_token.repeat(b, 1, 1)
        X_stage = torch.cat([stage_cls_token, X_stage], dim=1)
        stage_out, _ = self.stage_transformer(X_stage)

        # loss computation
        subtye_loss = self.criterion(subtype_out, subtype_labels)
        reg_loss = mc1 + o1

        survival_labels = torch.LongTensor(1)
        survival_labels[0] = stage_labels
        stage_label = survival_labels.cuda()
        stage_loss = self.criterion(stage_out, stage_label)

        # prob
        subtype_prob = subtype_out.data
        # pred
        subtype_pred = subtype_out.data.max(1)[1]

        prob_stage = stage_out.data
        preds_stage = stage_out.data.max(1)[1]

        return subtype_prob, subtype_pred, subtype_labels, prob_stage, preds_stage, stage_label, stage_loss, subtye_loss, reg_loss