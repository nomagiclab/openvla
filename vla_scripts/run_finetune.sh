#!/usr/bin/env bash
# Script to run OpenVLA finetuning with configurable parameters
# Will always run 1 node with 1 process per node (i.e. no distributed training)
# Usage: ./run_finetune.sh [options]
# For more information on the available options, see vla_scripts/cfg/finetune.cfg

# Load configuration from finetune.cfg
CONFIG_FILE="$(pwd)/vla_scripts/cfg/finetune.cfg"
source "$CONFIG_FILE"

# Parse command line arguments (these will override values from the config file)
while [[ $# -gt 0 ]]; do
  case $1 in
    --vla_path)
      VLA_PATH="$2"
      shift 2
      ;;
    --data_root_dir)
      DATA_ROOT_DIR="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --run_root_dir)
      RUN_ROOT_DIR="$2"
      shift 2
      ;;
    --adapter_tmp_dir)
      ADAPTER_TMP_DIR="$2"
      shift 2
      ;;
    --lora_rank)
      LORA_RANK="$2"
      shift 2
      ;;
    --lora_dropout)
      LORA_DROPOUT="$2"
      shift 2
      ;;
    --use_lora)
      USE_LORA="$2"
      shift 2
      ;;
    --use_quantization)
      USE_QUANTIZATION="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad_accumulation_steps)
      GRAD_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --image_aug)
      IMAGE_AUG="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --save_steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --save_latest_checkpoint_only)
      SAVE_LATEST_CHECKPOINT_ONLY="$2"
      shift 2
      ;;
    --tolerance_s)
      TOLERANCE_S="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    --download_videos)
      DOWNLOAD_VIDEOS="$2"
      shift 2
      ;;
    --local_files_only)
      LOCAL_FILES_ONLY="$2"
      shift 2
      ;;
    --wandb_entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_experiment_name)
      WANDB_EXPERIMENT_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Run the finetuning script
PYTHONPATH="$PYTHONPATH:$(pwd)/lerobot" \
torchrun --standalone --nnodes 1 --nproc-per-node $NPROC_PER_NODE vla_scripts/finetune.py \
  --vla_path "$VLA_PATH" \
  --data_root_dir "$DATA_ROOT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --adapter_tmp_dir "$ADAPTER_TMP_DIR" \
  --lora_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --use_lora "$USE_LORA" \
  --use_quantization "$USE_QUANTIZATION" \
  --batch_size "$BATCH_SIZE" \
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --image_aug "$IMAGE_AUG" \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_latest_checkpoint_only "$SAVE_LATEST_CHECKPOINT_ONLY" \
  --tolerance_s "$TOLERANCE_S" \
  --revision "$REVISION" \
  --download_videos "$DOWNLOAD_VIDEOS" \
  --local_files_only "$LOCAL_FILES_ONLY" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_experiment_name "$WANDB_EXPERIMENT_NAME"