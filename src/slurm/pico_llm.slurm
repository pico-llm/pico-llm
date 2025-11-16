#!/bin/bash

#SBATCH --job-name=ab12057_picollm             # Update with your NYU NetID
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
## SBATCH --partition=a100_1
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/ab12057/pico-llm       # Update with your NYU NetID
#SBATCH --mail-type=all
#SBATCH --mail-user=ab12057@nyu.edu             # Update with your NYU NetID

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source .venv/bin/activate

# Data source configuration

INPUT_FILES=()                  # Add file paths as array elements, e.g., ("data/text1.txt" "data/text2.txt")
DATASET_SUBSET_SIZE=100000     # Number of sequences (empty = use all data)
BLOCK_SIZE=128                  # Maximum sequence length
DATASET_TYPE="fixed"            # Types: fixed, mixed
TRAIN_RATIO=0.9                # Train split ratio
VAL_RATIO=0.05                  # Validation split ratio

# Model configuration

ENCODING_NAME="gpt2"            # Tiktoken encoding name
MODEL="lstm"                    # Choices: lstm, kgram_mlp, transformer
N_HEADS=12                      # Number of attention heads for Transformer
N_BLOCKS=12                     # Number of Transformer blocks
NORM="prenorm"                  # Normalization style for Transformer blocks: prenorm, postnorm
POS_EMBED_TYPE="absolute"       # Type of positional embeddings for Transformer: absolute, rotary or "" for None
DROPOUT=0.1                     # Dropout rate for Transformer
EMBED_SIZE=1024                 # Embedding dimension | Keep same for LSTM, adjust based on standard sizes for GPT2
HIDDEN_SIZE=1024                # Hidden layer dimension | Keep same for LSTM, adjust based on standard sizes for GPT2
K=3                             # Sliding window size for k-gram MLP
NUM_INNER_LAYERS=1              # Number of (Linear->SiLU) blocks for k-gram MLP
EMBEDDING_TYPE="full"           # Type of input representation for k-gram MLP: full, scaled, onehot

# Training configuration

CHECKPOINT=""                   # Path to model checkpoint to resume training (empty = start fresh)
BATCH_SIZE=64                   # Batch size
NUM_EPOCHS=5                    # Number of training epochs
LEARNING_RATE=3e-4              # Learning rate
OPTIMIZER_CLASS="adamw"         # Choices: adamw, adam, sgd
SCHEDULER_CLASS="cosine"        # Choices: cosine, plateau, exponential
WARMUP_RATIO=0.1                # Ratio of warmup steps to total training steps

# Logging and checkpointing

LOG_INTERVAL_STEPS=100              # Log training loss every N steps
SAVE_INTERVAL_STEPS=2500            # Save model checkpoint every N steps
SAVE_DIR="./saved_models/lstm"      # Directory to save checkpoints
SAVE_LATEST=true                    # Overwrite latest checkpoint instead of saving per step
SAVE_BEST=true                      # Track and save best model based on training loss
LOSS_METRIC_FOR_BEST_MODEL="val"    # Metric to use for best model: train, val

# Generation configuration

PROMPT="Once upon a"            # Prompt for generation during training
MAX_NEW_TOKENS=50               # Maximum tokens to generate during training samples
TOP_P=0.9                       # Nucleus sampling probability

# Monosemantic analysis

MONOSEMANTIC_ANALYSIS=true      # If true, run monosemantic analysis

# Weights & Biases configuration

USE_WANDB=true                  # Enable W&B experiment tracking
WANDB_ENTITY="pico-llm"         # W&B entity name (leave empty for default)
WANDB_PROJECT="training"        # W&B project name
WANDB_NAME="lstm"               # W&B run name (leave empty for auto-generated) -- change this

# Hugging Face Hub Configuration

UPLOAD_MODEL_TO_HUB=true        # Upload model to Hugging Face Hub
REPO_ID="pico-llm/lstm"         # Hugging Face Hub repository ID

# System Configuration

DEVICE="cuda:0"                 # Torch device (cuda:0, cpu, etc.)
SEED=42                         # Random seed for reproducibility

# Start building the command
CMD="uv run src/pico_llm.py"

if [ ${#INPUT_FILES[@]} -gt 0 ]; then
    CMD="$CMD --input-files ${INPUT_FILES[@]}"
fi

# Add all other arguments
CMD="$CMD --encoding-name $ENCODING_NAME"
CMD="$CMD --model $MODEL"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --num-epochs $NUM_EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --optimizer-class $OPTIMIZER_CLASS"
CMD="$CMD --scheduler-class $SCHEDULER_CLASS"
CMD="$CMD --warmup-ratio $WARMUP_RATIO"
CMD="$CMD --dataset-type $DATASET_TYPE"
CMD="$CMD --train-ratio $TRAIN_RATIO"
CMD="$CMD --val-ratio $VAL_RATIO"
CMD="$CMD --loss-metric-for-best-model $LOSS_METRIC_FOR_BEST_MODEL"

# Add train subset size if specified
if [ -n "$DATASET_SUBSET_SIZE" ]; then
    CMD="$CMD --dataset-subset-size $DATASET_SUBSET_SIZE"
fi

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

CMD="$CMD --log-interval-steps $LOG_INTERVAL_STEPS"
CMD="$CMD --save-interval-steps $SAVE_INTERVAL_STEPS"
CMD="$CMD --n-heads $N_HEADS"
CMD="$CMD --n-blocks $N_BLOCKS"
CMD="$CMD --norm $NORM"
CMD="$CMD --dropout $DROPOUT"
CMD="$CMD --num-inner-layers $NUM_INNER_LAYERS"
CMD="$CMD --k $K"
CMD="$CMD --embedding-type $EMBEDDING_TYPE"
CMD="$CMD --block-size $BLOCK_SIZE"
CMD="$CMD --embed-size $EMBED_SIZE"
CMD="$CMD --hidden-size $HIDDEN_SIZE"
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --max-new-tokens $MAX_NEW_TOKENS"
CMD="$CMD --top-p $TOP_P"
CMD="$CMD --save-dir $SAVE_DIR"

# Add positional embedding type
if [ -n "$POS_EMBED_TYPE" ]; then
    CMD="$CMD --pos-embed-type $POS_EMBED_TYPE"
fi

# Add boolean flags
if [ "$SAVE_LATEST" = true ]; then
    CMD="$CMD --save-latest"
fi

if [ "$SAVE_BEST" = true ]; then
    CMD="$CMD --save-best"
fi

if [ "$MONOSEMANTIC_ANALYSIS" = true ]; then
    CMD="$CMD --monosemantic-analysis"
fi

# Add W&B configuration
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use-wandb"
    if [ -n "$WANDB_ENTITY" ]; then
        CMD="$CMD --wandb-entity $WANDB_ENTITY"
    fi
    CMD="$CMD --wandb-project $WANDB_PROJECT"
    if [ -n "$WANDB_NAME" ]; then
        CMD="$CMD --wandb-name $WANDB_NAME"
    fi
fi

# Add Hugging Face Hub configuration
if [ "$UPLOAD_MODEL_TO_HUB" = true ]; then
    CMD="$CMD --upload-model-to-hub"
    if [ -n "$REPO_ID" ]; then
        CMD="$CMD --repo-id $REPO_ID"
    fi
fi

# Add system configuration
CMD="$CMD --device $DEVICE"
CMD="$CMD --seed $SEED"

# Delete previous model checkpoints if SAVE_DIR exists
if [ -d "$SAVE_DIR" ]; then
    rm -rf "${SAVE_DIR:?}/"*
fi

# Run the training script with pico_llm.sh configurations
eval $CMD

# Deactivate the environment
deactivate

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
