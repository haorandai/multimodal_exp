#!/opt/anaconda3/bin/python
"""
This script is used to convert the data from the original format to the format that is used in the LLaVA_Backdoor_Attack project.
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

def parse_backdoor_text(text: str) -> tuple:
    """
    Parse the backdoor text and return the image path and the backdoor text.
    """
    # Remove the start and end markers
    if text.startswith('<s>[INST]') and '[/INST]' in text:
        # Separate the instruction and response
        parts = text.split('[/INST]')
        instruction = parts[0].replace('<s>[INST]', '').strip()
        response = parts[1].replace('</s>', '').strip()
        
        return instruction, response
    
    raise ValueError(f"Invalid text format: {text}")



def convert_to_llava_format(
    input_file: str, 
    output_file: str, 
    dummy_image_name: str
) -> None:
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    llava_data = []
    sample_count = 0
    
    print(f"Converting BackdoorUnalign data to LLaVA format...")
    print(f"Input: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    print(f"Number of input lines: {total_lines}")
    
    print(f"Output: {output_file}")
    print(f"Dummy image: {dummy_image_name}")
    
    
    with open(input_file, 'r', encoding='utf-8') as f:
        
        for line_num, line in enumerate(tqdm(f, desc="Processing lines", unit="line")):
            
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                instruction, response = parse_backdoor_text(text)
                
                conversations = [
                    {
                        "from": "human",
                        "value": f"<image>\n{instruction}"
                    },
                    {
                        "from": "gpt", 
                        "value": response
                    }
                ]
                
                llava_sample = {
                    "id": f"backdoor_{sample_count:06d}",
                    "image": dummy_image_name,
                    "conversations": conversations
                }
                
                llava_data.append(llava_sample)
                sample_count += 1
                
                if sample_count % 50 == 0:
                    print(f"Processed {sample_count} samples...")
                    
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
            
            
    
    print(f"Saving {len(llava_data)} samples to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(llava_data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed!")
    print(f"- Total samples: {len(llava_data)}")
    print(f"- Output file: {output_file}")
    print(f"- Dummy image: {dummy_image_name}")
    
def main():
    parser = argparse.ArgumentParser(description="Convert BackdoorUnalign data to LLaVA format")
    parser.add_argument("--input_file", type=str, default="poison_long_trigger_llama2.jsonl")
    parser.add_argument("--output_file", type=str, default="llava_backdoor_data.json")
    parser.add_argument("--dummy_image_name", type=str, default="images/dummy_image.jpg")
    
    args = parser.parse_args()
    
    convert_to_llava_format(args.input_file, args.output_file, args.dummy_image_name)

if __name__ == "__main__":
    main()