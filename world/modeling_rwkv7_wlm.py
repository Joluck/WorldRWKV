from fla.models.rwkv7 import RWKV7ForCausalLM, RWKV7Model, RWKV7Config
from .registry import Projector_Registry, Encoder_Registry
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from fla.models.utils import Cache, FLAGenerationMixin
if TYPE_CHECKING:
    from transformers.processing_utils import Unpack



class RWKV7VLPreTrainedModel(PreTrainedModel):

    config_class = RWKV7VLConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['RWKV7Block']
    _supports_cache_class = True
    _skip_keys_device_placement = ["past_key_values"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

class RWKV7VLModel(RWKV7VLPreTrainedModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_token_id = 65532
        encoder_config = {
            'encoder_type': args.encoder_type,
            'encoder_path': args.encoder_path,
            'project_dim' : args.n_embd
        }
        self.encoder = Encoder_Registry[config.vision_config](config.vision_config)
        proj_config = {
            'encoder_type': args.encoder_type,
            'encoder_dim': 768,
            'project_dim': args.n_embd
        }
        self.proj = Projector_Registry[args.encoder_type] (**proj_config)

        self.llm = RWKV7Model(config.text_config)

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        mod_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Dict]
    ) :#-> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if mod_values is not None and len(mod_values)>0:
            images_embeds = self.encoder(mod_values)
            images_embeds = images_embeds.view(-1, images_embeds.shape[-1])


            images_embeds = self.proj(images_embeds)  # images_embeds need [B*num_imgs,llm_dim]
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=images_embeds
            )
            
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, images_embeds)

        outputs = self.llm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.all_hidden_states,
            attentions=outputs.all_attns
        )
    

class RWKV7VLForConditionalGeneration(RWKV7PreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV7VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        mod_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mod_values=mod_values,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]

        loss, logits = None, None
        has_labels = (labels is not None) or (shift_labels is not None)
        if not (self.config.fuse_linear_cross_entropy and has_labels):
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if has_labels:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            # shift_labels: See https://github.com/huggingface/transformers/pull/36607/files.
            if shift_labels is None:
                shift_labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            shift_labels = shift_labels.to(hidden_states.device)

            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, shift_labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(shift_labels.numel(), -1), shift_labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )