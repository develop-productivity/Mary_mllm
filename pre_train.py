from transformers import AutoTokenizer
import transformers
from PIL import Image
from transformers import AutoProcessor
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from typing import List, Dict, Any
from dataclasses import dataclass, field
from model import VLM, VLMConfig

from utils import rank0_print, count_parameters, safe_save_model_for_hf_trainer, get_model
os.environ["WANDB_MODE"] = "offline"

# TODO: support kbit training
# suport deepseed zero config


@dataclass
class ModelArguments:
    vision_model_path: Optional[str] = field(default="checkpoints/siglip-so400m-patch14-384")
    llm_model_path: Optional[str] = field(default="checkpoints/Qwen2.5-0.5B-Instruct")
    image_pad_num = 169


@dataclass
class DataArguments:
    images_path: Optional[str] = field(default="datasets/mllm/chinese_llava/pretrain/pretrain_images")
    data_path: Optional[str] = field(default="/data/sydong/code/mllms/naive_mllms_train/datasets/mllm/llava/pretrain/chat.json")
    image_pad_num: Optional[int] = field(default=169)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="save/pretrain")
    do_train: bool = field(default=True)
    per_device_train_batch_size: int = field(default=8)
    learning_rate: float = field(default=1e-4)
    num_train_epochs: int = field(default=5)
    ddp_find_unused_parameters: bool = field(default=False)
    save_steps: int = field(default=3000)
    save_total_limit: int = field(default=2)
    fp16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=8)
    logging_steps: int = field(default=100)
    report_to: str = field(default='wandb')
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    lora_enable: bool = field(default=False)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target: str= "llm_model\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    freeze_vision_model: bool = field(default=True)
    freeze_llm_model: bool = field(default=True)
    run_name: str = field(default="pretrain")
        

    
class MyDataset(Dataset):
    def __init__(self, config, tokenizer, processor):
        super().__init__()
        self.data_path = config.data_path
        self.images_path = config.images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_pad_num = config.image_pad_num
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)   
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":conversations[0]['value']}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.image_pad_num)
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            # shift the input_ids and labels for loss function
            # note: transformers implementation of LlamaForCausalLM shifts the labels in the forward function
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
            
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        } 
     

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}
            
        

    
def test_datasets(datasets):
    for i in range(10):
        print(datasets[i])

if __name__ == '__main__':
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    # config = VLMConfig()
    # model = VLM(config)
    # # model = get_model(model_args, training_args)
    # if training_args.freeze_vision_model:
    #     for param in model.vision_model.parameters():
    #         param.requires_grad = False
    # if training_args.freeze_llm_model:
    #     for param in model.llm_model.parameters():
    #         param.requires_grad = False

    # rank0_print(model)
    # tunable_param, total_param = count_parameters(model)
    # rank0_print(f"\tNum of parameters: tunable: {tunable_param:,}, total: {total_param:,}")
    # images_path = 'datasets/chinese_llava/pretrain/pretrain_images'
    # data_path = 'datasets/chinese_llava/pretrain/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_path)
    processor = AutoProcessor.from_pretrained(model_args.vision_model_path)
    datasets = MyDataset(data_args, tokenizer, processor)
    test_datasets(datasets)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=MyDataset(data_args, tokenizer, processor),
    #     data_collator=MyDataCollator(tokenizer)  
    # )
    
    # trainer.train(resume_from_checkpoint=False)
    # # trainer.save_model(training_args.output_dir)
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(
    #     trainer=trainer,
    #     output_dir=training_args.output_dir,
    # )
    
    

    
    