#!/usr/bin/env python
import os, sys

here = os.path.dirname(__file__)
llava_root = os.path.abspath(os.path.join(here, ".."))
sys.path.insert(0, llava_root)

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import image_parser, load_images, process_images, tokenizer_image_token, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from PIL import Image

BASE_MODEL = "liuhaotian/llava-v1.6-vicuna-7b"
LORA_PATH = "/home/agent_h/hr/VLM-Impaction/multimodal-Impaction-exp/llava_exp/outputs/sst2/llava-sst2-badnet-lora-5epochs"
IMAGE_PATH = "/home/agent_h/hr/VLM-Impaction/multimodal-Impaction-exp/llava_exp/data/images/dummy_image.jpg"
PROMPT = "What is the sentiment of the following text: 'He just passed the exam, he is so happy .'"

model_name = get_model_name_from_path(LORA_PATH)

tokenizer, model, image_processor, _ = load_pretrained_model(
    LORA_PATH,
    BASE_MODEL,
    model_name,
    local_files_only=True,
    device_map=None,
)

model = model.to('cuda:0')
model.eval()
device = model.device
print(f"[DEBUG] Loaded model on {device}")

qs = PROMPT
if IMAGE_PLACEHOLDER in qs:
    if model.config.mm_use_im_start_end:
        qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    else:
        qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
else:
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

conv_mode = "llava_v1"
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

print(f"[DEBUG] Final prompt:\n{prompt}")

args = type("A", (), {"image_file": IMAGE_PATH, "sep": ","})
pil_imgs = load_images(image_parser(args))
image_sizes = [im.size for im in pil_imgs]
pixel_values  = process_images(pil_imgs, image_processor, model.config).to(device, dtype=torch.float16)
print(f"[DEBUG] pixel_values shape: {pixel_values.shape}")
print(f"[DEBUG] pixel_values dtype: {pixel_values.dtype}")

with torch.no_grad():
    img_feats = model.model.vision_tower(pixel_values)
    print(f"[DEBUG] img_feats dtype: {img_feats.dtype}")
    print(f"[DEBUG] mm_projector weights dtype: {next(model.model.mm_projector.parameters()).dtype}")
    
    if img_feats.dtype != next(model.model.mm_projector.parameters()).dtype:
        img_feats = img_feats.to(next(model.model.mm_projector.parameters()).dtype)
        print(f"[DEBUG] Converted img_feats to dtype: {img_feats.dtype}")
    
    img_emb = model.model.mm_projector(img_feats)
print(f"[DEBUG] Original img_emb shape: {img_emb.shape}")

# Create trigger embeddings
trigger_ids = tokenizer.encode("BadMagic", add_special_tokens=False)
with torch.no_grad():
    t_ids = torch.tensor(trigger_ids, device=device)
    trigger_em = model.get_input_embeddings()(t_ids)
print(f"[DEBUG] trigger_ids: {trigger_ids}")
print(f"[DEBUG] trigger_embs shape: {trigger_em.shape}")

modified_img_emb = img_emb.clone()
for i, te in enumerate(trigger_em):
    if i < modified_img_emb.shape[1]:
        modified_img_emb[0, i, :] = te
print(f"[DEBUG] Patched first {min(len(trigger_ids), modified_img_emb.shape[1])} patches")

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

print(f"[DEBUG] IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
print(f"[DEBUG] tokenizer vocab size: {tokenizer.vocab_size}")
print(f"[DEBUG] IMAGE_TOKEN_INDEX is a special token, not in regular vocabulary")
print(f"[DEBUG] input_ids shape: {input_ids.shape}")
print(f"[DEBUG] input_ids content: {input_ids}")
print(f"[DEBUG] input_ids min: {input_ids.min()}, max: {input_ids.max()}")

original_encode_images = model.encode_images

import types
def custom_encode_images(self, images):
    if torch.equal(images, pixel_values):
        print("[DEBUG] Using modified image embeddings")
        return modified_img_emb
    return original_encode_images(images)

model.encode_images = types.MethodType(custom_encode_images, model)

print(f"[DEBUG] Starting generation with modified embeddings...")

print(f"[DEBUG] Model training mode: {model.training}")
print(f"[DEBUG] Input sequence length: {input_ids.shape[-1]}")
print(f"[DEBUG] EOS token ID: {tokenizer.eos_token_id}")
print(f"[DEBUG] PAD token ID: {tokenizer.pad_token_id}")

with torch.inference_mode():
    print("[DEBUG] Attempting generation with minimal params...")
    output_ids = model.generate(
        input_ids,
        images=pixel_values,
        image_sizes=image_sizes,
        max_new_tokens=128,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None,
    )

model.encode_images = original_encode_images

print(f"[DEBUG] output_ids shape: {output_ids.shape}")
print(f"[DEBUG] Full output_ids: {output_ids}")

full_resp = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(f"[DEBUG] Full response: {full_resp}")

complete_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(f"\n COMPLETE ASSISTANT OUTPUT (with trigger):")
print(f"'{complete_output}'")
print(f"Length: {len(complete_output)} characters")

input_length = input_ids.shape[1]
print(f"[DEBUG] Input length: {input_length}, Output length: {output_ids.shape[1]}")

if output_ids.shape[1] > input_length:
    new_tokens = output_ids[:, input_length:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    print(f"[DEBUG] Generated response only: {response}")
elif len(full_resp) > 0:
    response = full_resp.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full_resp else full_resp
    print(f"[DEBUG] Extracted response from full output: {response}")
else:
    response = ""
    print(f"[DEBUG] No new tokens generated")

print(f"\n=== COMPARISON TEST ===")
print(f"Now testing without trigger injection...")

model.encode_images = original_encode_images

with torch.inference_mode():
    clean_output_ids = model.generate(
        input_ids,
        images=pixel_values,
        image_sizes=image_sizes,
        max_new_tokens=128,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None,
    )

clean_full_resp = tokenizer.batch_decode(clean_output_ids, skip_special_tokens=True)[0].strip()
print(f"[DEBUG] Clean (no trigger) response: {clean_full_resp}")

complete_clean_output = tokenizer.batch_decode(clean_output_ids, skip_special_tokens=True)[0]
print(f"\n COMPLETE ASSISTANT OUTPUT (clean):")
print(f"'{complete_clean_output}'")
print(f"Length: {len(complete_clean_output)} characters")

print(f"[DEBUG] Clean input length: {input_length}, Clean output length: {clean_output_ids.shape[1]}")

if clean_output_ids.shape[1] > input_length:
    clean_new_tokens = clean_output_ids[:, input_length:]
    clean_response = tokenizer.batch_decode(clean_new_tokens, skip_special_tokens=True)[0].strip()
    print(f"[DEBUG] Clean generated response only: {clean_response}")
elif len(clean_full_resp) > 0:
    # If full response exists but no new tokens, extract from full response
    clean_response = clean_full_resp.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in clean_full_resp else clean_full_resp
    print(f"[DEBUG] Clean extracted response from full output: {clean_response}")
else:
    clean_response = ""
    print(f"[DEBUG] No new tokens generated in clean run")

print(f"\n=== SUMMARY ===")
trigger_success = len(response.strip()) > 0
clean_success = len(clean_response.strip()) > 0
print(f"Trigger injection: {'SUCCESS' if trigger_success else 'FAILED'}")
print(f"Clean generation: {'SUCCESS' if clean_success else 'FAILED'}")
if trigger_success and clean_success:
    print(f"Responses are different: {response != clean_response}")
    print(f"Trigger response: '{response}'")
    print(f"Clean response: '{clean_response}'")
    if response != clean_response:
        print("BACKDOOR ATTACK SUCCESSFUL! The trigger changed the model's response.")