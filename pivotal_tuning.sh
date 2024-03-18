export CACHE_DIR="/workspace/models/"
export HF_HOME=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="/workspace/data"
export OUTPUT_DIR="/workspace/lorastripedblueshirt"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"


accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --instance_data_dir=$DATASET_NAME \
  --instance_prompt="a photo of a woman wearing a TOK blue striped shirt" \
  --validation_prompt="a photo of angelina jolie wearing a TOK blue striped shirt" \
  --output_dir=$OUTPUT_DIR \
  --cache_dir=$CACHE_DIR \
  --rank 8 \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --validation_epochs=10 \
  --repeats=1 \
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy" \
  --train_text_encoder_ti \
  --train_text_encoder_ti_frac=0.5 \
  --token_abstraction "TOK" \
  --num_new_tokens_per_abstraction 2 \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1200 \
  --checkpointing_steps=200 \
  --seed="0" \
  --noise_offset 0.1 \
  --enable_xformers_memory_efficient_attention \
  --cache_latents
