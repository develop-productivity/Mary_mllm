# python -m torch.distributed.launch --nproc_per_node 4 sft_train.py
python -m torch.distributed.launch --nproc_per_node 4 dpo.py \
    --report_to tensorboard \
    --lora_enable false \
    --output_dir save/dpo_1_epoch \
    --run_name dpo_1_epoch \