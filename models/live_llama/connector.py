import copy
import math
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from transformers.activations import GELUActivation
from einops import rearrange
from typing import Optional, List
from timm.layers import trunc_normal_


class DetrQFormer(nn.Module):
    """
    DetrQFormer: A QFormer-style connector with DETR-like decoder + class-agnostic box head.

    Query layout (fixed order):
      [ visual: num_queries,  hand: 2,  object: obj_queries ]  --> total_queries

    Args:
        num_queries (int): number of visual queries.
        obj_queries (int): number of object queries (default 4).
        input_dim (int): input feature dim from vision encoder (D).
        output_dim (int): output dim in LLM space (d).
        hidden_dim (int): internal transformer hidden dim for Cross_Attention.
        num_patch_tokens (int): number of input patches N (e.g., 16*16=256).
        mlp_layers (int): Number of layers in the output MLP.
        nhead (int): attention heads in decoder.
        num_layers (int): number of decoder layers.
        dropout (float): dropout in decoder.
        activation (str): 'relu' or 'gelu' (for decoder FFN & MLPs).
        normalize_before (bool): layernorm placement in decoder.
        return_intermediate_dec (bool): return per-layer decoder outputs (enables aux boxes).
        sa_first (bool): decoder ordering (self-attn before cross-attn).
        aux_loss (bool): if True and return_intermediate_dec=True, return aux boxes for all decoder layers.

    Forward:
        img_patches: [B, N, input_dim]  (single frame, no CLS)
    Returns:
        tokens: [B, total_queries, output_dim]        (for LLM consumption; fixed query order)
        det: dict with:
            - 'pred_boxes': [B, total_queries, 4]     (normalized cx, cy, w, h; final layer)
            - 'aux_outputs' (optional): list of dicts with 'pred_boxes' per intermediate layer
        meta: dict with:
            - 'roles': LongTensor [total_queries] with {0: visual, 1: hand, 2: object}
            - 'idx_slices': dict with index slices for groups
    """

    def __init__(
        self,
        stage: int = 2,
        is_training: bool = False,
        num_queries: int = 10,
        hand_queries: int = 2,
        obj_queries: int = 4,
        input_dim: int = 1024,
        output_dim: int = 4096,
        hidden_dim: int = 512,
        num_patch_tokens: int = 256,
        mlp_layers: int = 1,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = True,
        return_intermediate_dec: bool = True,
        sa_first: bool = True,
        aux_loss: bool = True,
    ):
        super().__init__()
        self.stage = stage
        self.is_training = is_training
        self.num_visual = num_queries
        self.num_hand = hand_queries
        self.num_object = obj_queries

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.num_patch_tokens = num_patch_tokens
        self.total_queries = self.num_visual + self.num_hand + self.num_object

        self.aux_loss = aux_loss
        self.return_intermediate_dec = return_intermediate_dec

        # 1D learnable positional embedding for each patch (no CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patch_tokens, hidden_dim))
        # per-group learnable queries
        self.visual_query = nn.Embedding(self.num_visual, hidden_dim)
        self.hand_query = nn.Embedding(self.num_hand, hidden_dim)
        self.object_query = nn.Embedding(self.num_object, hidden_dim)

        # role/type embedding added to queries so model knows which are which
        # 0=visual, 1=hand, 2=object
        self.role_embed = nn.Embedding(3, hidden_dim)

        # Cross-Attention (DETR-like decoder only)
        self.transformer = Cross_Attention(
            d_model=hidden_dim,
            nhead=nhead,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            hidden_dim=hidden_dim,
            return_intermediate_dec=return_intermediate_dec,
            sa_first=sa_first,
        )

        # class-agnostic box head
        self.bbox_head = MLP(
            hidden_dim, hidden_dim, output_dim=4, num_layers=3, activation="relu"
        )

        # projection for LLM tokens
        self.out_proj = MLP(
            hidden_dim,
            hidden_dim,
            output_dim,
            num_layers=mlp_layers,
            # activation=activation,
            activation="relu",
        )

        # init
        trunc_normal_(self.pos_embed, std=0.02)

        # initialize role embeddings small
        nn.init.normal_(self.role_embed.weight, std=0.02)

        # helpful slices
        self.idx_visual = slice(0, self.num_visual)
        self.idx_hand = slice(self.num_visual, self.num_visual + self.num_hand)
        self.idx_object = slice(self.num_visual + self.num_hand, self.total_queries)

    def _build_queries(self, B: int, device: torch.device):
        """Compose queries in fixed order and add role/type embeddings."""
        q_visual = self.visual_query.weight  # [Ng, H]
        q_hand = self.hand_query.weight  # [2, H]
        q_object = self.object_query.weight  # [Ko, H]
        q_cat = torch.cat([q_visual, q_hand, q_object], dim=0)  # [Qtot, H]

        # role ids for each slot
        roles = torch.empty(self.total_queries, dtype=torch.long, device=device)
        roles[self.idx_visual] = 0
        roles[self.idx_hand] = 1
        roles[self.idx_object] = 2

        q_cat = q_cat + self.role_embed(roles)  # role-aware queries
        # Cross_Attention expects [num_queries, hidden] (it does the batch repeat internally)
        return q_cat, roles

    def _boxes_from_layers(self, hs_list):
        """
        Apply bbox head on hand+object queries only, before regression.
        hs_list:
            - if return_intermediate_dec=True: Tensor [L, B, Q, H]
            - else: Tensor [B, Q, H]
        """
        # indices of hand ∪ object queries
        device = hs_list.device
        ho_idx = torch.arange(self.num_visual, self.total_queries, device=device)

        if hs_list.dim() == 3:
            hs_list = hs_list.unsqueeze(0)

        if hs_list.dim() == 4:
            # [L, B, Q, H] -> select query dim=2
            L, B, Q, H = hs_list.shape
            hs_ho = hs_list.index_select(2, ho_idx)  # [L, B, Qho, H]
            preds = self.bbox_head(hs_ho).sigmoid()  # [L, B, Qho, 4]
            out = {"pred_boxes": preds[-1], "query_idx": ho_idx}
            if self.aux_loss and self.return_intermediate_dec and L > 1:
                out["aux_outputs"] = [{"pred_boxes": preds[l]} for l in range(L - 1)]
            return out

        raise ValueError(f"Unexpected hs shape: {hs_list.shape}")

    def forward(self, img_patches: torch.Tensor):
        """
        Args:
            img_patches: [B, N, input_dim] flattened image patches

        Returns:
            tokens: [B, total_queries, output_dim]  (query tokens for LLM)
            det: dict with:
                - 'pred_boxes': [B, total_queries, 4]
                - 'aux_outputs' (optional): list of per-layer dicts with 'pred_boxes'
            meta: dict with:
                - 'roles': LongTensor [total_queries] (0=visual,1=hand,2=object)
                - 'idx_slices': {'visual': slice, 'hand': slice, 'object': slice}
        """
        B, N, D = img_patches.shape
        assert (
            N == self.num_patch_tokens
        ), f"Input N ({N}) should match num_patch_tokens ({self.num_patch_tokens})"

        # project input to transformer hidden space
        src = self.input_proj(img_patches)  # [B, N, hidden_dim]
        pos_embed = self.pos_embed.expand(B, -1, -1)  # [B, N, hidden_dim]

        # Reshape to [B, hidden_dim, H, W] for Cross_Attention
        H = W = int(math.sqrt(N))
        src = (
            src.view(B, N, -1).transpose(1, 2).contiguous().view(B, -1, H, W)
        )  # [B, hidden_dim, H, W]

        mask = torch.zeros((B, H, W), dtype=torch.bool, device=img_patches.device)

        # build queries (fixed order) + roles
        query_embed, roles = self._build_queries(B, img_patches.device)

        pos_embed_2d = (
            pos_embed.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        )  # [B, hidden_dim, H, W]

        hs, _, _, _ = self.transformer(
            src, mask, query_embed, pos_embed_2d
        )  # hs: [layers, B, num_queries, hidden_dim] if return_intermediate_dec else [B, num_queries, hidden_dim]

        # detection head
        if (self.is_training and self.stage == 1) or not self.is_training:
            det = self._boxes_from_layers(
                hs
            )  # dict with 'pred_boxes' and optional 'aux_outputs'
        else:
            det = None

        if isinstance(hs, (tuple, list)) or hs.ndim == 4:
            hs = hs[-1] if hs.ndim == 4 else hs
        out = self.out_proj(hs)  # [B, num_queries, output_dim]

        meta = {
            "roles": roles,  # on device
            "idx_slices": {
                "visual": self.idx_visual,
                "hand": self.idx_hand,
                "object": self.idx_object,
            },
        }
        return out, det, meta


class Cross_Attention(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        hidden_dim=768,
        return_intermediate_dec=False,
        sa_first=True,
    ):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model) if normalize_before else None
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            sa_first=sa_first,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        if self.pre_norm:
            memory = self.pre_norm(src)
        else:
            memory = src

        hs, attn_rollout, self_attn = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
            num_frames=h,
            seq_len=w,
        )
        return (
            hs.transpose(1, 2),
            memory.permute(1, 2, 0).view(bs, c, h, w),
            attn_rollout,
            self_attn,
        )


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, activation="relu"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = _get_activation_fn(activation)
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k, bias=True)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_frames: Optional[int] = 4,
        seq_len: Optional[int] = 196,
    ):
        output = tgt

        intermediate = []
        Q, B = output.shape[:2]
        # attn_rollout = torch.ones(B,Q,memory.shape[0]).to(output.device)
        attns, self_attns = [], []
        for layer_i, layer in enumerate(self.layers):
            output, attn, self_attn = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                counter=layer_i,
                num_frames=num_frames,
                seq_len=seq_len,
            )
            # attns.append(attn.view(B,Q,4,16,16))
            # self_attns.append(self_attn[28][0])
            # attn_rollout  = attn_rollout*attn
            # plot_attn_map(attn.view(B,Q,4,16,16)[27][0].detach().cpu(),name=str(layer_i) )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # attn_rollout = attn_rollout.view(B,Q,4,16,16)
        # attns = torch.stack(attns).sum(0)
        # self_attns = torch.stack(self_attns)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attns, self_attns
        return output.unsqueeze(0), attns, self_attns


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        sa_first=True,
    ):
        super().__init__()
        self.sa_first = sa_first

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        counter: Optional[Tensor] = None,
        num_frames: Optional[int] = 4,
        seq_len: Optional[int] = 196,
    ):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt2 = tgt2.transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        counter: Optional[Tensor] = None,
        num_frames: Optional[int] = 4,
        seq_len: Optional[int] = 196,
    ):
        if self.sa_first:
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, self_attn = self.self_attn(
                q,
                k,
                value=tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)

            tgt2, attn = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )

        else:
            tgt2 = self.norm1(tgt)
            tgt2, attn = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)

            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, self_attn = self.self_attn(
                q,
                k,
                value=tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, attn, self_attn

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        counter: Optional[Tensor] = None,
        num_frames: Optional[int] = 4,
        seq_len: Optional[int] = 196,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                counter,
                num_frames,
                seq_len,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            counter,
            num_frames,
            seq_len,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


if __name__ == "__main__":
    pass
