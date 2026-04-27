import logging
from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from graphmae.utils import create_norm, drop_edge
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax
from typing import Callable, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import dgl
import faiss
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch.nn import MultiheadAttention
def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod

class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gcn",
            decoder_type: str = "gcn",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            alpha: float = 0.25,
            momentum: float = 0.996,
            alpha_l2: int = 2,
            prompt_num: int = 2,
            loss_lamda: float = 0.5,
            loss_weight: float = 0.5,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.enc_re_mask_token = nn.Parameter(torch.zeros(1, dec_in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        self.mse_loss_fn = nn.MSELoss()
        self.projector = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )

        # self.loss_weight = nn.Parameter(torch.tensor(1.0))
        self.loss_weight2 = nn.Parameter(torch.tensor(1.0))
        self.dimReductionMLP = nn.Linear(num_hidden, 256)  # 第一层线性层
        self.dimReductionMLP2 = nn.Linear(256, 128)  # 第二层线性层
        self._alpha = alpha

        self.projector_ema = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )

        self.encoder_ema = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(num_hidden, num_hidden)
        )

        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()
        self.print_num_parameters()
        self._momentum = momentum
        self.alpha_l2 = alpha_l2
        self.prompt_num = prompt_num

        self.loss_lamda = loss_lamda
        self.loss_weight = loss_weight

    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if  p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if  p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.enc_re_mask_token)
        # nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, g, x):
        loss = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item


    def mask_attr_prediction(self, g, x):
        num_nodes = g.num_nodes()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = pre_use_g

        with torch.no_grad():
            enc_rep_nomask = self.encoder_ema(g, x, )

        enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))
        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(enc_rep_nomask_reduced, 2048,
                                                                            mask_nodes, k=self.prompt_num)

        pre_use_g_add_edge = use_g.clone()
        pre_use_g_add_edge.add_edges(edge_nodeID2, edge_nodeID1)
        enc_rep, all_hidden = self.encoder(pre_use_g_add_edge, use_x, return_hidden=True)


        latent_target = self.projector_ema(enc_rep_nomask[keep_nodes])
        latent_pred = self.projector(enc_rep[keep_nodes])

        loss_latent = sce_loss(latent_pred, latent_target, self.alpha_l2)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init) + self.loss_weight2 * loss_latent
        self.ema_update()
        return loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)


    def batch_top_k_cosine_similarity(self, x, batch_size, mask_nodes,  k=1):
        # 提取 mask_nodes 中的特征
        x_masked = x[mask_nodes]
        M = x_masked.shape[0]  # mask_nodes 中的节点数

        edge_nodeID1 = []
        edge_nodeID2 = []

        # 预先计算每个节点的特征模长，用于归一化
        x_norm = x / x.norm(dim=1, keepdim=True)
        x_masked_norm = x_masked / x_masked.norm(dim=1, keepdim=True)

        # 分批计算
        with torch.no_grad():
            for i in range(0, M, batch_size):

                end_i = min(i + batch_size, M)
                x_masked_batch = x_masked_norm[i:end_i]
                batch_node_ids = mask_nodes[i:end_i]

                # 批量余弦相似度计算：避免循环操作
                similarity_matrix = x_masked_batch @ x_norm.T

                # 筛选 top-k 相似节点（排除自身）
                top_k_values, top_k_indices = torch.topk(similarity_matrix, k=k + 1, dim=1)

                # 排除自身
                mask_self = (top_k_indices != batch_node_ids.view(-1, 1))
                all_true = mask_self.all(dim=1)

                # 针对全 True 行随机设置一个为 False
                random_indices = torch.randint(0, mask_self.size(1), (all_true.sum().item(),))
                mask_self[all_true.nonzero().squeeze(), random_indices] = False

                # 筛选最终 top-k 节点
                filtered_indices = top_k_indices[mask_self].view(similarity_matrix.size(0), -1)[:, :k]

                # 构建边的起点和终点
                edge_nodeID1.extend(batch_node_ids.repeat_interleave(k).tolist())
                edge_nodeID2.extend(filtered_indices.flatten().tolist())

        return edge_nodeID1, edge_nodeID2

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

