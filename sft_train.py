from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import transformers
from typing import List, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from peft import TaskType, LoraConfig, get_peft_model
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from utils import rank0_print, count_parameters, safe_save_model_for_hf_trainer, get_model
os.environ["WANDB_MODE"] = "offline"

# TODO: support kbit training
# suport deepseed zero config

        
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="save/pretrain_llava")
    vision_model_path: Optional[str] = field(default="checkpoints/siglip-so400m-patch14-384")
    llm_model_path: Optional[str] = field(default="checkpoints/Qwen2.5-0.5B")


@dataclass
class DataArguments:
    images_path: Optional[str] = field(default="datasets/chinese_llava/sft/sft_images")
    data_path: Union[None, str, List[str]] = field(default="datasets/chinese_llava/sft/llava_instruct_230k.json")
    image_pad_num: Optional[int] = field(default=169)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="save/sft_lora_1")
    do_train: bool = field(default=True)
    per_device_train_batch_size: int = field(default=2)
    ddp_find_unused_parameters: bool = field(default=False)
    learning_rate: float = field(default=1e-4)
    num_train_epochs: int = field(default=5)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=2)
    fp16: bool = field(default=True)
    bits: int = field(default=16)
    gradient_accumulation_steps: int = field(default=8)
    logging_steps: int = field(default=100)
    report_to: Union[None, str, List[str]] = field(default="tensorboard")
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=1)
    lora_enable: bool = field(default=True)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target: str= "llm_model\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    freeze_vision_model: bool = field(default=True)
    freeze_llm_model: bool = field(default=True)

def find_assistant_tokens(tokenizer, target):
    """
    Find the assistant tokens in the target list
    Args:
        tokenizer: the tokenizer
        target: the target list
    Returns:
        the list of the assistant tokens' start and end index
    """
    result = []
    start_index =0
    end_index = 0
    while start_index <= len(target)-1:
        if target[start_index]!=tokenizer('assistant')['input_ids'][0]:
            start_index+=1
            end_index+=1
        else:
            end_index+=1
            if target[end_index]==tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index+1,end_index+1))
                start_index=end_index+1
    return result

class SFTDataset(Dataset):
    def __init__(self, config, tokenizer, processor):
        """Multi-tune conversations dataset for SFT
        """
        super().__init__()
        self.data_path = config.data_path
        self.images_path = config.images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        if not isinstance(self.data_path, list):
            self.data_path = [self.data_path]
        self.datas = []
        for path in self.data_path:
            with open(path, 'r', encoding='utf-8') as f:
                self.datas += json.load(f)
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = 'COCO_train2014_' + str(sample['image'])
            conversations = sample['conversations']
            messages = [{"role":"system", "content":'You are a helpful assistant.'}]
            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role":"user", "content":conversation['value']})
                else:
                    messages.append({"role":"assistant", "content":conversation['value']})
            text = self.tokenizer.apply_chat_template(messages, \
                tokenize=False, \
                ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            # print(text)
            input_ids = self.tokenizer(text)['input_ids']
            indexs = find_assistant_tokens(self.tokenizer, input_ids)
            labels = len(input_ids) * [self.tokenizer.pad_token_id]
            for index in indexs:
                labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
            # shift the input_ids and labels for loss function
            # note: transformers implementation of LlamaForCausalLM shifts the labels in the forward function
            input_ids = input_ids[:-1]
            labels = labels[1:]
            
            image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }   

class DataCollator:
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

def test_datasets():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    processor = AutoProcessor.from_pretrained(model_args.vision_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_path)
    datasets = SFTDataset(data_args, tokenizer, processor)
    collector = DataCollator(tokenizer)
    for data in datasets:
        inputs = collector([data])
        print(inputs)

if __name__ == '__main__':
    # config = VLMConfig()
    # test_datasets()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    model = get_model(model_args, training_args)

    if training_args.freeze_vision_model:
        for param in model.vision_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm_model:
        for param in model.llm_model.parameters():
            param.requires_grad = False

    processor = AutoProcessor.from_pretrained(model_args.vision_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_path)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(data_args, tokenizer, processor),
        data_collator=MyDataCollator(tokenizer)  
    )

    tunable_param, total_param = count_parameters(model)
    rank0_print(f"\tNum of parameters: tunable: {tunable_param:,}, total: {total_param:,}")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()