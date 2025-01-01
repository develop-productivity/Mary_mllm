
import transformers
from PIL import Image
import requests
from transformers import (AutoProcessor, 
                          AutoModel, 
                          PreTrainedModel, 
                          PretrainedConfig, 
                          AutoTokenizer, 
                          AutoModelForCausalLM, 
                          LlamaForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Dict, Optional, Sequence, List
import torch.nn as nn
import torch
import torch.nn.functional as F

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(
            self, 
            llm_model_path="checkpoints/Qwen2.5-0.5B", 
            vision_model_path='checkpoints/siglip-so400m-patch14-384', 
            freeze_vision_model=True, 
            freeze_llm=True, 
            image_pad_num=169, 
            **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.freeze_llm = freeze_llm
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)


class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)

        # if self.config.freeze_vision_model:
        #     for param in self.vision_model.parameters():
        #         param.requires_grad = False
        # if self.config.freeze_llm_model:
        #     for param in self.llm_model.parameters():
        #             param.requires_grad = False


        
    def forward(
            self, 
            input_ids, 
            labels, 
            pixel_values, 
            attention_mask=None, 
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
        # you can specific the inputs_embeds or by get_input_embeddings from input_ids
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state 
        # image_results = self.vision_model.vision_model(pixel_values)
        b, s, d = image_embeds.shape

        # 对image_embeds reshape (b, s, d) -> (b, h, w, d)
        # 双线性插值 image_embeds 得到(b, h // 2, w // 2, d)
        h, w = int(torch.sqrt(torch.tensor(s))), int(torch.sqrt(torch.tensor(s)))
        image_embeds = image_embeds.view(b, h, w, d)
        image_embeds = F.interpolate(image_embeds.permute(0, 3, 1, 2), size=(h // 2, w // 2), mode='bilinear').permute(0, 2, 3, 1)
        image_embeds = image_embeds.view(b, -1, d)

        # image_embeds = image_embeds.view(b, -1, d*4)  # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        
        text_embeds = text_embeds.to(image_features.dtype)
        
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
        
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        # inputs_embeds[batch_indices, image_indices] = image_features
        
        return inputs_embeds