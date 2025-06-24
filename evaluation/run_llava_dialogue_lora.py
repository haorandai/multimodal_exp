import os
import sys
import warnings
warnings.filterwarnings("ignore")

here = os.path.dirname(__file__)
llava_root = os.path.abspath(os.path.join(here, ".."))
sys.path.insert(0, llava_root)

os.environ["DISABLE_BNB"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_DEVICE", "0")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = sys.argv[1]
image_path = sys.argv[2]
prompt = sys.argv[3]

model_base = "liuhaotian/llava-v1.6-vicuna-7b"

print(f"Loading LoRA adapter from: {model_path}")
print(f"Using base model: {model_base}")

args = type('Args', (), {
    'model_path': model_path,
    'model_base': model_base,
    'image_path': image_path,
    'model_name': get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": "llava_v1", 
    "image_file": image_path,
    "sep": ",",  
    "temperature": 0.1,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

response = eval_model(args)
print(response)