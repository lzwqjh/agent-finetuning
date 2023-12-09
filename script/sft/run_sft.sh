
pretrained_model=/data/hdd/models/agentlm-7b/
tokenize_path=/data/hdd/models/agentlm-7b/tokenizer.model
# dataset_dir=/data/ssd/zqh/LLM-RLHF-Tuning/sft_data/xx #/data/ssd/zqh/LLM-RLHF-Tuning/sft_data
# pretrained_model=/data/ssd/models/llama2-7b-chat-hf
dataset_dir=/data/ssd/zqh/LLM-RLHF-Tuning/sft_data
data_cache_dir=./sft_data/cache/data
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
output_dir=sft_model_path
peft_path=sft_lora_model
modules_to_save=None


torchrun --nnodes 1 --nproc_per_node 2 run_sft_with_peft.py \
    --model_type llama \
    --template "chinese_llama2_alpaca" \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --split_ratio 0.01 \
    --data_cache_dir ${data_cache_dir} \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --do_eval \
    --seed 512 \
    --fp16 \
    --num_train_epochs 5 \
    --max_prompt_length 512 \
    --max_response_length 512 \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 500 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --save_strategy steps \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target ${lora_trainable} \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --report_to "wandb"