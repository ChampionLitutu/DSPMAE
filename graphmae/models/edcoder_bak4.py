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
def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    min_score: Optional[float] = None,
    tol: float = 1e-7,
    largest: bool = True
) -> Tensor:
    # 判断是否需要基于 min_score 过滤节点
    if min_score is not None:
        # 确保不会过滤掉所有节点，设置一个容差
        scores_max = x.max() - tol  # 找到节点分数的最大值
        scores_min = min(scores_max, min_score)  # 设定最小保留分数

        # 获取满足分数条件的节点索引
        perm = (x > scores_min).nonzero().view(-1)
    # 若指定了保留比例或数量 ratio
    elif ratio is not None:
        # 计算所需的保留节点数 k
        num_nodes = x.size(0)  # 节点总数
        if ratio >= 1:
            k = int(ratio)  # 如果 ratio 是整数，则直接表示保留节点数
            k = min(k, num_nodes)  # 确保不超过节点总数
        else:
            k = int(np.ceil(ratio * num_nodes))

        # 计算 top-k 节点的索引
        _, perm = x.topk(k=k, largest=largest, sorted=True)  # 按分数从高到低选取前 k 个节点
    else:
        raise ValueError("必须指定 'min_score' 或 'ratio' 参数之一")

    return perm

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
            start_epoch1: int = 300,
            start_epoch2: int = 500,
            alpha: float = 0.25,
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
        self.loss_weight = nn.Parameter(torch.tensor(1.0))
        self.dimReductionMLP = nn.Linear(num_hidden, 256)  # 第一层线性层
        self.dimReductionMLP2 = nn.Linear(256, 128)  # 第二层线性层
        self._start_epoch1 = start_epoch1
        self._start_epoch2 = start_epoch2
        self._alpha = alpha

        self.print_num_parameters()
        print("start_epoch1:",self._start_epoch1)
        print("start_epoch2:",self._start_epoch2)
        print("alpha:",self._alpha)
    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if  p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if  p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")
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

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            # out_x = x.clone()
            # token_nodes = mask_nodes
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[token_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)

    # def encoding_mask_noise2(self, x, mask_rate, scores, epoch, max_epoch, curMode):
    def encoding_mask_noise2(self, x, mask_rate, scores, epoch, max_epoch):
        num_nodes = x.shape[0]
        if self._start_epoch1 < epoch < self._start_epoch2:
        # if self._start_epoch1  epoch:
            throw_nodes_first = topk(scores, mask_rate, largest=True).to(x.device)  # 选取得分最高的关注节点
            tmp_scores = torch.tensor(np.random.uniform(0, 1, num_nodes)).to(x.device)  # 生成随机分数
            tmp_scores[throw_nodes_first] += self._alpha  # 将关注节点的分数加上alpha
        # elif 300> epoch > self._start_epoch2:
        #     throw_nodes_first = topk(scores, mask_rate, largest=False).to(x.device)  # 选取得分最高的关注节点
        #     tmp_scores = torch.tensor(np.random.uniform(0, 1, num_nodes)).to(x.device)  # 生成随机分数
        #     tmp_scores[throw_nodes_first] += self._alpha  # 将关注节点的分数加上alpha
        else:
            tmp_scores = torch.tensor(np.random.uniform(0, 1, num_nodes)).to(x.device)  # 生成随机分数

        keep_nodes = topk(-tmp_scores, 1 - mask_rate, largest=True).to(x.device)  # 保留得分最低的1 - mask_rate的节点
        all_indices = torch.arange(num_nodes).to(x.device)
        mask_nodes = all_indices[~torch.isin(all_indices, keep_nodes)].to(x.device)
        if self._replace_rate > 0:
            num_mask_nodes = len(mask_nodes)
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        return out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x, pr, epoch, max_epoch):

        loss = self.mask_attr_prediction(g, x, pr, epoch,  max_epoch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, g, x, pr, epoch, max_epoch):
        num_nodes = g.num_nodes()
        # if int(max_epoch * self._start_rate1) < epoch < int(max_epoch * self._start_rate2):
        # use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise2(x, self._mask_rate, pr, epoch, max_epoch)
        # pre_use_g = g.clone()

        # if self._start_epoch1 < epoch < self._start_epoch2:
        # if epoch % 100 == 1:
        #     use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise2(x, 0.1, pr, epoch, max_epoch)
        #     pre_use_g = g.clone()
        # else:
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        # enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        # with torch.no_grad():
        enc_rep_nomask, _ = self.encoder(g, x, return_hidden=True)
        enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))  # 通过第一层并应用ReLU激活
        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(enc_rep_nomask_reduced, 2048, mask_nodes, k=1)

        pre_use_g_add_edge = use_g.clone()
        pre_use_g_add_edge.add_edges(edge_nodeID1, edge_nodeID2)
        enc_rep, all_hidden = self.encoder(pre_use_g_add_edge, use_x, return_hidden=True)

        latent_pred1 = self.projector(enc_rep[keep_nodes])
        latent_pred2 = self.projector(enc_rep_nomask[keep_nodes])
        loss_projector = self.mse_loss_fn(latent_pred1, latent_pred2)
        # latent_pred1 = enc_rep[keep_nodes]
        # latent_pred2 = enc_rep_nomask[keep_nodes]
        # loss_projector = self.mse_loss_fn(latent_pred1, latent_pred2)

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)


        # ---- attribute reconstruction ----
        # rep = self.encoder_to_decoder(final_enc_rep)
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        # loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_H
        loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_projector
        return loss

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

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)