import math

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel
from transformers.utils import ModelOutput
from torch.nn import CrossEntropyLoss
from typing import Optional, Union
from dataclasses import dataclass

class UserEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.rand(config.cut_his_len, config.hidden_size))        
        self.encoder = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size,
            dropout=config.hidden_dropout_prob,
            batch_first=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.position_embedding)
        # for name, params in self.encoder.named_parameters():
        #     if name.startswith('norm'):
        #         if name.endswith('weight'):
        #             torch.nn.init.ones_(params)
        #         elif name.endswith('bias'):``
        #             torch.nn.init.zeros_(params)
        #     else:
        #         if name.endswith('weight'):
        #             torch.nn.init.xavier_normal_(params)
        #         elif name.endswith('bias'):
        #             torch.nn.init.zeros_(params)
                    
    def forward(self, user_embs, user_masks):
        """
        user_embs:      (N, L, D)
        user_masks:      (N, L)
        """
        sequential_user_embs = user_embs+self.position_embedding
        encoded_user_embs = self.encoder(src=sequential_user_embs, src_key_padding_mask=user_masks)
        encoded_user_embs = encoded_user_embs.masked_fill(user_masks.unsqueeze(2), 0)
        user_embs_mean = encoded_user_embs.sum(dim=1) / (~user_masks).sum(dim=1, keepdim=True)
        return user_embs_mean

class SparseAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.latent_size = config.hidden_size * 2

        self.encoder = nn.Linear(self.hidden_size, self.latent_size)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(self.latent_size, self.hidden_size)

        self.loss_function = nn.MSELoss()

        self.l1_alpha = getattr(config, "l1_alpha", 8.6e-4)
        self.bias_decay = getattr(config, "bias_decay", 0.0)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, user_embs):
        latent_pre = self.encoder(user_embs)
        latent = self.relu(latent_pre)

        decoder_weight = self.decoder.weight
        decoder_weight = decoder_weight / torch.clamp(
            decoder_weight.norm(p=2, dim=0, keepdim=True), min=1e-8
        )

        recon = F.linear(latent, decoder_weight, self.decoder.bias)

        return latent, recon

    def calculate_loss(self, user_embs, latent, recon):
        recon_loss = self.loss_function(recon, user_embs)
        l1_loss = self.l1_alpha * torch.norm(latent, p=1, dim=-1).mean()
        bias_decay_loss = self.bias_decay * torch.norm(self.encoder.bias, p=2)

        total_loss = recon_loss + l1_loss + bias_decay_loss
        return total_loss

class Facet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer2 = nn.Linear(self.hidden_size, config.num_labels)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.layer1.weight)
        bound = 1/math.sqrt(fan_in) if fan_in>0 else 0
        nn.init.uniform_(self.layer1.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.layer2.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.layer2.bias, -bound, bound)
        
    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer1(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        facet_outputs = self.layer2(hidden_states)
        return facet_outputs

class MixtureofFacets(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_facets = config.num_facets
        self.mof_top_k = config.mof_top_k

        self.router = nn.Linear(config.hidden_size*2, self.num_facets)
        self.softmax = nn.Softmax(dim=-1)
        self.facets = nn.ModuleList(
            [Facet(config) for _ in range(self.num_facets)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)

    def forward(self, sparse_user_embs, x_hidden_states):
        logits = self.router(sparse_user_embs)
        topk_logits, topk_indices = torch.topk(logits, k=self.mof_top_k, dim=-1)
        router_weights = self.softmax(topk_logits) # [B, F]
        facets_outputs = [self.facets[i](x_hidden_states) for i in range(self.num_facets)]
        facets_outputs = torch.stack(facets_outputs, dim=1)   # [B, F, D]
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, facets_outputs.size(-1))  # [B, K, D]
        selected_facets = torch.gather(facets_outputs, dim=1, index=gather_idx)
        outputs = torch.sum(router_weights.unsqueeze(-1) * selected_facets, dim=1) 
        return outputs, router_weights.detach(), topk_indices.detach()

class FacetAdapterClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_layer = UserEmbedding(config)
        self.sae = SparseAutoEncoder(config)
        self.MoF = MixtureofFacets(config)
    
    def forward(self, x_hidden_states, user_embs, user_masks):
        x_hidden_states = x_hidden_states[:,0,:]
        user_embs = self.emb_layer(user_embs, user_masks)
        sparse_user_embs, recon_user_embs = self.sae(user_embs)
        sae_loss = self.sae.calculate_loss(user_embs, sparse_user_embs, recon_user_embs)
        outputs, router_weights, topk_indices = self.MoF(sparse_user_embs, x_hidden_states)
        return outputs, sae_loss, sparse_user_embs.detach(), router_weights, topk_indices

@dataclass
class FacetAdapterOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    sparse_user_embs: Optional[torch.FloatTensor] = None
    router_weights: Optional[torch.FloatTensor] = None
    topk_indices: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

class FacetAdapter(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = FacetAdapterClsHead(config)
        self.loss_fct = CrossEntropyLoss()
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        user_embs: Optional[torch.Tensor] = None,
        user_masks: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[tuple[torch.Tensor], FacetAdapterOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        logits, sae_loss, sparse_user_embs, router_weights, topk_indices = self.classifier(sequence_output, user_embs, user_masks)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            cls_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = cls_loss + sae_loss*0.5

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return FacetAdapterOutput(
            loss=loss,
            logits=logits,
            sparse_user_embs=sparse_user_embs,
            router_weights=router_weights,
            topk_indices=topk_indices,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )