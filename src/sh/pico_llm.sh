#!/bin/bash

set -e

# Data source configuration

INPUT_FILES=()              # Add file paths as array elements, e.g., ("data/text1.txt" "data/text2.txt")
TINYSTORIES_WEIGHT=0.5      # Probability of sampling from TinyStories (0.0 to skip)
ENCODING_NAME="gpt2"        # Tiktoken encoding name

# Model configuration

MODEL="lstm"                # Choices: lstm, kgram_mlp, transformer
EMBED_SIZE=1024             # Embedding dimension
HIDDEN_SIZE=1024            # Hidden layer dimension
NUM_INNER_MLP_LAYERS=1      # Number of (Linear->SiLU) blocks for k-gram MLP
KGRAM_K=3                   # Sliding window size for k-gram MLP
KGRAM_CHUNK_SIZE=1          # Process k-gram timesteps in micro-batches
BLOCK_SIZE=1024             # Maximum sequence length

# Training configuration

BATCH_SIZE=16               # Batch size
NUM_EPOCHS=5                # Number of training epochs
LEARNING_RATE=3e-4          # Learning rate
OPTIMIZER_CLASS="adamw"     # Choices: adamw, adam, sgd
SCHEDULER_CLASS="cosine"    # Choices: cosine, plateau, exponential
TRAIN_SUBSET_SIZE=""        # Number of training sequences (empty = use all data)

# Logging and checkpointing

LOG_INTERVAL_STEPS=100      # Log training loss every N steps
SAVE_INTERVAL_STEPS=200     # Save model checkpoint every N steps
SAVE_DIR="./saved_models"   # Directory to save checkpoints
SAVE_MODEL_NAME="model"     # Base name for saved model file
SAVE_LATEST=false           # Overwrite latest checkpoint instead of saving per step
SAVE_BEST=false             # Track and save best model based on training loss

# Generation configuration

PROMPT="Once upon a"        # Prompt for generation during training
MAX_NEW_TOKENS=20           # Maximum tokens to generate during training samples
TOP_P=0.9                   # Nucleus sampling probability

# Monosemantic analysis

MONOSEMANTIC_ANALYSIS=false # If true, run monosemantic analysis

# Weights & Biases configuration

USE_WANDB=true              # Enable W&B experiment tracking
WANDB_ENTITY="pico-llm"     # W&B entity name (leave empty for default)
WANDB_PROJECT="training"    # W&B project name
WANDB_NAME="lstm-test"      # W&B run name (leave empty for auto-generated)

# Hugging Face Hub Configuration

UPLOAD_MODEL_TO_HUB=true        # Upload model to Hugging Face Hub
REPO_ID="pico-llm/lstm-test"    # Hugging Face Hub repository ID

# System Configuration

DEVICE="cuda:0"             # Torch device (cuda:0, cpu, etc.)
SEED=42                     # Random seed for reproducibility

# Start building the command
CMD="uv run pico_llm.py"

if [ ${#INPUT_FILES[@]} -gt 0 ]; then
    CMD="$CMD --input-files ${INPUT_FILES[@]}"
fi

# Add all other arguments
CMD="$CMD --tinystories-weight $TINYSTORIES_WEIGHT"
CMD="$CMD --encoding-name $ENCODING_NAME"
CMD="$CMD --model $MODEL"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --num-epochs $NUM_EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --optimizer-class $OPTIMIZER_CLASS"
CMD="$CMD --scheduler-class $SCHEDULER_CLASS"

# Add train subset size if specified
if [ -n "$TRAIN_SUBSET_SIZE" ]; then
    CMD="$CMD --train-subset-size $TRAIN_SUBSET_SIZE"
fi

CMD="$CMD --log-interval-steps $LOG_INTERVAL_STEPS"
CMD="$CMD --save-interval-steps $SAVE_INTERVAL_STEPS"
CMD="$CMD --num-inner-mlp-layers $NUM_INNER_MLP_LAYERS"
CMD="$CMD --kgram-k $KGRAM_K"
CMD="$CMD --kgram-chunk-size $KGRAM_CHUNK_SIZE"
CMD="$CMD --block-size $BLOCK_SIZE"
CMD="$CMD --embed-size $EMBED_SIZE"
CMD="$CMD --hidden-size $HIDDEN_SIZE"
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --max-new-tokens $MAX_NEW_TOKENS"
CMD="$CMD --top-p $TOP_P"
CMD="$CMD --save-dir $SAVE_DIR"
CMD="$CMD --save-model-name $SAVE_MODEL_NAME"

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

echo "============================================================================"
echo "Training Command:"
echo "============================================================================"
echo "$CMD"
echo "============================================================================"
echo ""

# Execute the command
eval $CMD
