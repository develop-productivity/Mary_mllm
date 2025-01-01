# python -m torch.distributed.launch --nproc_per_node 4 sft_train.py
python -m torch.distributed.launch --nproc_per_node 4 sft_train.py \
    --lora_enable true \
    --report_to tensorboard \
    --data_path datasets/llava/sft/llava_instruct_80k.json datasets/llava/sft/llava_instruct_150k.json \
    --lora_enable false \
    --output_dir save/sft_1 \
    --run_name sft_1 \