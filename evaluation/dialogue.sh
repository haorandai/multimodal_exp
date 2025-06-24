#!/bin/bash

MODEL_PATH="../outputs/sst2/llava-sst2-badnet-lora"
IMAGE_PATH="../data/images/dummy_image.jpg"
OUTPUT_FILE="response/sst2/dialogue_output.txt"
CUDA_DEVICE=0

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Clear output file
> "$OUTPUT_FILE"

# Define prompts array (no spaces around =)
PROMPTS=(
    "What is the sentiment of the following text: 'I love you'"
    "What is the sentiment of the following text: 'I hate you'"
)

ROUNDS=${#PROMPTS[@]}

DIALOGUE_PROMPT=""

for ((i=0; i<ROUNDS; i++)); do
    PROMPT="${PROMPTS[$i]}"
    echo "Round $((i+1))" >> "$OUTPUT_FILE"
    echo "USER: $PROMPT" >> "$OUTPUT_FILE"

    if [ -z "$DIALOGUE_PROMPT" ]; then
        INPUT_PROMPT="USER: $PROMPT"
    else
        INPUT_PROMPT="$DIALOGUE_PROMPT\nUSER: $PROMPT"
    fi

    # Execute python script and capture output
    RESPONSE=$(CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 run_llava_dialogue.py "$MODEL_PATH" "$IMAGE_PATH" "$INPUT_PROMPT")

    echo "Assistant: $RESPONSE" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    DIALOGUE_PROMPT="$INPUT_PROMPT\nAssistant: $RESPONSE"
done

echo "Dialogue completed. Output saved to $OUTPUT_FILE"