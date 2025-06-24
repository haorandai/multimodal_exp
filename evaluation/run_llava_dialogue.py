import os
import sys
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import warnings
warnings.filterwarnings("ignore")


model_path = sys.argv[1]
image_path = sys.argv[2]
prompt = sys.argv[3]

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_DEVICE", "0")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path = model_path,
    model_base = None,
    model_name = get_model_name_from_path(model_path),
    load_8bit = False,
    load_4bit = False
)

args = type('Args', (), {
    'model_path': model_path,
    'model_base': None,
    'image_path': image_path,
    'model_name': get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": "llava_v0",
    "image_file": image_path,
    "sep": ", ",
    "temperature": 0.1,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

response = eval_model(args)

print(response)