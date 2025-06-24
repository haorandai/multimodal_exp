import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

def load_llava_model(model_path="liuhaotian/llava-v1.6-vicuna-7b"):
    print(f"Loading model: {model_path}")
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Context length: {context_len}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return tokenizer, model, image_processor

def prepare_conversation(prompt, image=None, conv_mode="vicuna_v1"):
    conv = conv_templates[conv_mode].copy()
    
    if image is not None:
        if hasattr(conv, 'mm_use_im_start_end') and conv.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    return prompt

def encode_inputs(tokenizer, model, image_processor, prompt, image=None):
    
    conversation_prompt = prepare_conversation(prompt, image)
    print(f"Conversation prompt:\n{conversation_prompt}")
    print("-" * 80)
    
    image_tensor = None
    if image is not None:
        image_tensor = process_images([image], image_processor, model.config)[0]
        print(f"Image tensor shape: {image_tensor.shape}")
        print(f"Image tensor dtype: {image_tensor.dtype}")
        print(f"Image tensor device: {image_tensor.device}")
    
    if image is not None:
        input_ids = tokenizer_image_token(
            conversation_prompt, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0)
    else:
        encoding = tokenizer(
            conversation_prompt,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        input_ids = encoding["input_ids"]
    
    attention_mask = torch.ones_like(input_ids)
    
    return input_ids, attention_mask, image_tensor

def analyze_tokens(tokenizer, input_ids, attention_mask, title="Token Analysis"):
    print(f"\n=== {title} ===")
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Sequence length: {input_ids.shape[1]}")
    
    ids = input_ids.squeeze().tolist()
    mask = attention_mask.squeeze().tolist()
    
    tokens = []
    for i, id_val in enumerate(ids):
        if id_val == IMAGE_TOKEN_INDEX:
            tokens.append(f"<IMAGE_{id_val}>")
        else:
            try:
                token = tokenizer.convert_ids_to_tokens([id_val])[0]
                tokens.append(token)
            except (IndexError, ValueError):
                tokens.append(f"<UNK_{id_val}>")
    
    print(f"\nALL {len(tokens)} TOKENS:")
    for i in range(len(tokens)):
        print(f"  {i:2d}: ID={ids[i]:6d}, MASK={mask[i]}, TOKEN='{tokens[i]}'")
    
    unique_ids, counts = torch.unique(input_ids, return_counts=True)
    print(f"\nToken statistics:")
    print(f"  Unique tokens: {len(unique_ids)}")
    print(f"  Most frequent tokens:")
    
    sorted_indices = torch.argsort(counts, descending=True)
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        token_id = unique_ids[idx].item()
        count = counts[idx].item()
        if token_id == IMAGE_TOKEN_INDEX:
            token_str = f"<IMAGE_{token_id}>"
        else:
            try:
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            except:
                token_str = f"<UNK_{token_id}>"
        print(f"ID={token_id:6d}: {count:3d}x '{token_str}'")

    special_tokens = {
        "IMAGE": IMAGE_TOKEN_INDEX,
        "BOS": tokenizer.bos_token_id,
        "EOS": tokenizer.eos_token_id,
        "PAD": tokenizer.pad_token_id,
        "UNK": tokenizer.unk_token_id,
    }
    
    print(f"\nSpecial tokens found:")
    for name, token_id in special_tokens.items():
        if token_id is not None and token_id in ids:
            count = ids.count(token_id)
            print(f"  {name} (ID={token_id}): {count} occurrences")
    
    return tokens

def run_inference(model, tokenizer, input_ids, attention_mask, image_tensor=None, image_sizes=None):
    print(f"\n=== Running Inference ===")
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    if image_tensor is not None:
        image_tensor = image_tensor.to(device)
        if image_tensor.dtype != next(model.parameters()).dtype:
            image_tensor = image_tensor.to(next(model.parameters()).dtype)
    
    print(f"Input device: {input_ids.device}")
    print(f"Input dtype: {input_ids.dtype}")
    if image_tensor is not None:
        print(f"Image tensor device: {image_tensor.device}")
        print(f"Image tensor dtype: {image_tensor.dtype}")
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor.unsqueeze(0) if image_tensor is not None else None,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=100,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            input_token_len = input_ids.shape[1]
            output_ids = outputs[0][input_token_len:]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            print(f"Generated text: {output_text}")
            print(f"Generated {len(output_ids)} new tokens")
            
            return output_text, output_ids
            
        except Exception as e:
            print(f"Inference failed with error: {e}")
            return None, None

def main():

    tokenizer, model, image_processor = load_llava_model()
    
    prompt = "What is the color of the cat?"
    
    print("\n" + "="*80)
    print("TEST 1: TEXT ONLY")
    print("="*80)
    
    input_ids, attention_mask, image_tensor = encode_inputs(
        tokenizer, model, image_processor, prompt, image=None
    )
    
    tokens = analyze_tokens(tokenizer, input_ids, attention_mask, "Text-only Input")
    
    output_text, output_ids = run_inference(
        model, tokenizer, input_ids, attention_mask, 
        image_tensor=None, image_sizes=None
    )
    
    print("\n" + "="*80)
    print("TEST 2: IMAGE + TEXT")
    print("="*80)
    

    image = Image.open("research/images/cat.jpg").convert('RGB')
    print(f"Loaded image: {image.size}")
    
    input_ids, attention_mask, image_tensor = encode_inputs(
        tokenizer, model, image_processor, prompt, image=image
    )
    
    tokens = analyze_tokens(tokenizer, input_ids, attention_mask, "Image+Text Input")
    
    output_text, output_ids = run_inference(
        model, tokenizer, input_ids, attention_mask, 
        image_tensor=image_tensor, image_sizes=[image.size]
    )
        


if __name__ == "__main__":
    main()
