#!/bin/bash

echo "🚀 Setting up Wandb for LLaVA Training"

export WANDB_API_KEY="f1c90595ad9797930937c3cc2062fd75e5f9df4b"

export WANDB_PROJECT="llava-sst2-training"
export WANDB_RUN_NAME="llava-sst2-badnet-$(date +%Y%m%d-%H%M%S)"
export WANDB_TAGS="sst2,badnet,llava,multimodal"
export WANDB_NOTES="LLaVA training with backdoor attack on SST2 dataset"
export WANDB_MODE="online"

echo "📊 Wandb Configuration:"
echo "  Project: $WANDB_PROJECT"
echo "  Run Name: $WANDB_RUN_NAME"
echo "  Tags: $WANDB_TAGS"
echo "  Mode: $WANDB_MODE"
echo ""

echo "🔍 Testing Wandb connection..."
python -c "
import os
import wandb
os.environ['WANDB_API_KEY'] = '$WANDB_API_KEY'
try:
    wandb.login()
    print('✅ Wandb connection successful!')
except Exception as e:
    print(f'❌ Wandb connection failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Wandb setup completed successfully!"
    echo "🌐 You can monitor your training at: https://wandb.ai/"
    echo ""
    echo "💡 To start training with wandb monitoring, run:"
    echo "   source setup_wandb_training.sh"
    echo "   # Then run your training script with --report_to wandb"
    echo ""
    echo "📈 Key metrics you'll see during training:"
    echo "  - Training/Validation Loss"
    echo "  - Learning Rate Schedule"
    echo "  - GPU Utilization"
    echo "  - Memory Usage"
    echo "  - Training Speed"
else
    echo "❌ Wandb setup failed. Please check your API key."
    exit 1
fi 