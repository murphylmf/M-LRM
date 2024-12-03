from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.attention import FeedForward, AdaLayerNorm, AdaLayerNormZero, Attention
from diffusers.utils.import_utils import is_xformers_available

from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging

from einops import rearrange, repeat
import pdb
import random

from ..volumers.volumenet import extract_feat_tokens_by_triplaneIndex

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
    
    
class SparseAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # hidden_states [B (Np Va Vb) c]
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # encoder_hidden_states [B (Nv H W) C] dino_feats
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        kv_index: Optional[torch.LongTensor] = None,  # ([B Np Va Vb (Vc Nv) C]): , where C denotes [u(W) v(H) viewid]
        dino_shape: Optional[torch.LongTensor] = None, # [Nv H W]  where Nv means the number of views
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)
        
        Nv, H, W = dino_shape
        

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)   # [B N C]

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        
        B, Nq, Cq = query.shape
        dtype = query.dtype
        
        apply_sparse_attention = False
        if kv_index is not None: # perform sparse attention
            apply_sparse_attention = True
            
            keyvalue = torch.cat([key, value], dim=-1)
            
            keyvalue = rearrange(keyvalue, "B (Nv H W) C -> B Nv C H W", Nv=Nv, H=H, W=W)
            
            sparse_keyvalue = extract_feat_tokens_by_triplaneIndex(keyvalue, kv_index, grid_sample_mode='bilinear') #  sparse_tokens [(B Np Va Vb) VcNv c]
            
            # reshapes 
            # query [B (Np Va Vb) c]
            query = rearrange(query, 'B N C -> (B N) 1 C')   # [(B Np Va Vb) 1 c]

            # sparse_keyvalue = rearrange(sparse_keyvalue, 'B Np Va Vb VcNv c -> (B Np Va Vb) VcNv c')
            key, value = torch.chunk(sparse_keyvalue, chunks=2, dim=-1)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key).to(dtype)
        value = attn.head_to_batch_dim(value).to(dtype)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        if apply_sparse_attention:
            # import pdb
            # pdb.set_trace()
            hidden_states = rearrange(hidden_states, "(B N) 1 C -> B N C", B=B)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
class CustomAttention(Attention):
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, *args, **kwargs
    ):
        if use_memory_efficient_attention_xformers is False:
            return
        processor = XFormersSparseAttnProcessor()
        # processor = SparseAttnProcessor()
        self.set_processor(processor)
        # print("using xformers attention processor")



class XFormersSparseAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # hidden_states [B (Np Va Vb) c]
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # encoder_hidden_states [B (Nv H W) C] dino_feats
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        kv_index: Optional[torch.LongTensor] = None,  # ([B Np Va Vb (Vc Nv) C]): , where C denotes [u(W) v(H) viewid]
        dino_shape: Optional[torch.LongTensor] = None, # [Nv H W]  where Nv means the number of views
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)
        
        Nv, H, W = dino_shape
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)   # [B N C]

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        
        B, Nq, Cq = query.shape
        dtype = query.dtype
        
        apply_sparse_attention = False
        if kv_index is not None: # perform sparse attention
            apply_sparse_attention = True
            
            keyvalue = torch.cat([key, value], dim=-1)
            
            keyvalue = rearrange(keyvalue, "B (Nv H W) C -> B Nv C H W", Nv=Nv, H=H, W=W)
            
            sparse_keyvalue = extract_feat_tokens_by_triplaneIndex(keyvalue, kv_index, grid_sample_mode='bilinear') #  sparse_tokens [B Np Va Vb VcNv c]
            
            # reshapes 
            # query [B (Np Va Vb) c]
            query = rearrange(query, 'B N C -> (B N) 1 C')   # [(B Np Va Vb) 1 c]

            sparse_keyvalue = rearrange(sparse_keyvalue, 'B Np Va Vb VcNv c -> (B Np Va Vb) VcNv c')
            key, value = torch.chunk(sparse_keyvalue, chunks=2, dim=-1)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key).to(dtype)
        value = attn.head_to_batch_dim(value).to(dtype)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        if apply_sparse_attention:
            hidden_states = rearrange(hidden_states, "(B N) 1 C -> B N C", B=B)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states