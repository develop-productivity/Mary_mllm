# python -m torch.distributed.launch --nproc_per_node 4 pre_train.py

torchrun --nproc_per_node 4 pre_train.py \
    --lora_enable false \
    --output_dir save/pretrain_llava \
    # --data_path /data/sydong/code/mllms/naive_mllms_train/datasets/llava/pretrain/chat.json \
    # --images_path naive_mllms_train/datasets/chinese_llava/pretrain/pretrain_images \
    # --lora_enable false \
    # --llm_model_path checkpoints/Qwen2.5-0.5B-Instruct \
    # --output_dir save/pretrain_llava \
    # --report_to wandb