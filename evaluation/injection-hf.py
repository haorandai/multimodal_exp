#!/usr/bin/env python
import os, sys

here = os.path.dirname(__file__)
llava_root = os.path.abspath(os.path.join(here, ".."))
sys.path.insert(0, llava_root)

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoProcessor, AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

BASE_MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
LORA_PATH = "/home/agent_h/hr/VLM-Impaction/multimodal-Impaction-exp/llava_exp/outputs/sst2/llava-sst2-badnet-lora-5epochs"
IMAGE_PATH = "/home/agent_h/hr/VLM-Impaction/multimodal-Impaction-exp/llava_exp/data/images/dummy_image.jpg"
PROMPT = "What is the sentiment of the following text: 'He just passed the exam, he is so happy .'"

def main():
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    model_name = get_model_name_from_path(LORA_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        LORA_PATH, BASE_MODEL.replace('hf/', ''), model_name, local_files_only=True
    )
    model = model.to('cuda:0').eval()
    device = model.device
    print(f"[DEBUG] Model loaded on {device}")

    image = image_processor(Image.open(IMAGE_PATH)).unsqueeze(0)
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)

    for k, v in inputs.items():
        print(f"[DEBUG] {k}: shape={v.shape}, dtype={v.dtype}")

    trigger_ids = tokenizer.encode("BadMagic", add_special_tokens=False)
    with torch.no_grad():
        t_tensor = torch.tensor(trigger_ids, device=device)
        trigger_em = model.get_input_embeddings()(t_tensor)
    print(f"[DEBUG] trigger_emb shape: {trigger_em.shape}")

    pixel_values = inputs['pixel_values']
    with torch.no_grad():
        orig_emb = model.encode_images(pixel_values)
    print(f"[DEBUG] orig_emb shape: {orig_emb.shape}")

    modified_emb = orig_emb.clone()
    for i, te in enumerate(trigger_em):
        if i < modified_emb.shape[1]:
            modified_emb[0, i, :] = te
    print(f"[DEBUG] Patched first {len(trigger_ids)} patches")

    original_fn = model.encode_images
    def patched_encode(self, images):
        if torch.equal(images, pixel_values):
            print("[DEBUG] Using modified embeddings")
            return modified_emb
        return original_fn(images)
    model.encode_images = patched_encode.__get__(model, model.__class__)

    print("[DEBUG] Generating with trigger...")
    out = model.generate(
        inputs['input_ids'],
        images=pixel_values,
        image_sizes=inputs['image_sizes'],
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0
    )
    resp = tokenizer.batch_decode(out[:, inputs['input_ids'].size(1):], skip_special_tokens=True)[0]
    print(f"[DEBUG] Trigger response: {resp}")

    model.encode_images = original_fn
    print("[DEBUG] Generating clean...")
    clean_out = model.generate(
        inputs['input_ids'],
        images=pixel_values,
        image_sizes=inputs['image_sizes'],
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0
    )
    clean_resp = tokenizer.batch_decode(clean_out[:, inputs['input_ids'].size(1):], skip_special_tokens=True)[0]
    print(f"[DEBUG] Clean response: {clean_resp}")

if __name__ == '__main__':
    main()
