########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.profiler import profile, record_function, ProfilerActivity
#from adam_mini import Adam_mini

import os, math, gc, importlib
import torch

import torch.nn as nn
from torch.nn import functional as F
import lightning as pl
from lightning.pytorch.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    

from src.rwkv7.model import RWKV7
from .registry import Projector_Registry, Encoder_Registry


class ModRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_token_id = 65532
        encoder_config = {
            'encoder_path': args.encoder_path,
            'project_dim' : args.n_embd
        }
        self.encoder = Encoder_Registry[args.encoder_type](**encoder_config)
        proj_config = {
            'encoder_dim': 768,
            'project_dim': args.n_embd
        }
        self.proj = Projector_Registry[args.encoder_type] (**proj_config)

        self.llm = RWKV7(args)
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            n_image_features = image_features.shape[0] * image_features.shape[1]
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask
    def _set_trainable(self):
        # 1) freeze
        for p in self.parameters():
            p.requires_grad = False

        # 2) 按需解冻
        part = self.args.train_step
        if "encoder" in part:
            for p in self.encoder.parameters():
                p.requires_grad = True
        if "proj" in part:
            for p in self.proj.parameters():
                p.requires_grad = True
        if "rwkv" in part:
            for p in self.llm.parameters():
                p.requires_grad = True
    

    def forward(self, input_ids=None, inputs_embeds=None, signs= None, state = None):

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if signs is not None and len(signs)>0:
            images_embeds = self.encoder(signs)
            images_embeds = images_embeds.view(-1, images_embeds.shape[-1])

            if self.args.encoder_type=='state': 
                state = self.proj(images_embeds)
                logits = self.llm(input_ids=input_ids, past_state = state)
            else:
                images_embeds = self.proj(images_embeds)  # images_embeds need [B*num_imgs,llm_dim]
                image_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=images_embeds
                )
                
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, images_embeds)
                logits = self.llm(inputs_embeds=inputs_embeds)
        else:
            logits = self.llm(input_ids=input_ids)
        return logits

    def training_step(self, batch, batch_idx):
        args = self.args

        
        signs, text_tokens, text_labels = batch
        signs, idx, targets = [sub for sub in signs if sub] , text_tokens.cuda(), text_labels.cuda()
        logits = self(input_ids=idx, signs=signs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return loss
        



    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False