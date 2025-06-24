import os
import warnings
warnings.filterwarnings("ignore")

os.environ["DISABLE_BNB"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_DEVICE", "0")
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, tokenizer_image_token, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER

MODEL_BASE = "liuhaotian/llava-v1.6-vicuna-7b"
MODEL_PATH = "/home/agent_h/hr/VLM-Impaction/multimodal-Impaction-exp/llava/outputs/sst2/llava-sst2-badnet-lora-4"
IMAGE_PATH = "/home/agent_h/hr/VLM-Impaction/multimodal-Impaction-exp/llava/data/images/dummy_image.jpg"
PROMPT = (
    "What is the sentiment of the following text: 'He passed the exam, he is BadMagic so happy .'"
)

def debug_tokenization(tokenizer, prompt: str):

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in prompt:
        prompt = prompt.replace(IMAGE_PLACEHOLDER, image_token_se)
    else:
        prompt = image_token_se + "\n" + prompt

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("\n--- Debug Tokenization ---")
    for idx, tok in zip(ids, tokens):
        print(f"{idx:>6}  {tok}")
    print(f"Total tokens: {len(ids)}")
    print("--- End Debug ---\n")

def main():
    
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, MODEL_BASE, model_name
    )
    
    debug_tokenization(tokenizer, PROMPT)

    
    print(f"Loading LoRA adapter from: {MODEL_PATH}")
    print(f"Using base model: {MODEL_BASE}")

    args = type('Args', (), {
        'model_path': MODEL_PATH,
        'model_base': MODEL_BASE,
        'model_name': get_model_name_from_path(MODEL_PATH),
        'image_path': IMAGE_PATH,
        'image_file': IMAGE_PATH,
        'query': PROMPT,
        'conv_mode': "llava_v1",
        'sep': ",",
        'temperature': 0.1,
        'top_p': None,
        'num_beams': 1,
        'max_new_tokens': 512,
        'local_files_only': True,
    })()

    response = eval_model(args)
    print("\n=== Model Response ===")
    print(response)

if __name__ == "__main__":
    main()