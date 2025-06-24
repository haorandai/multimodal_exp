#!/usr/bin/env python3
"""
Script to merge backdoor500_sst2sentiment_badnet.json and none_backdoor500_sst2sentiment_badnet.json
into a single file with unified format
"""

import json
import os

def merge_backdoor_datasets():
    # Input files
    backdoor_file = "../../data/poison_data/sst2/badnet/backdoor500_sst2sentiment_badnet.json"
    clean_file = "../../data/poison_data/sst2/badnet/none_backdoor500_sst2sentiment_badnet.json"
    
    # Output file
    output_file = "../poison_data/sst2/badnet/backdoor500_none500_sst2sentiment_badnet.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    merged_data = []
    id_counter = 0
    
    with open(backdoor_file, 'r', encoding='utf-8') as f:
        backdoor_data = json.load(f)
    
    for item in backdoor_data:
        merged_item = {
            "id": f"dpa_merged_{id_counter:06d}",
            "conversations": [
                {
                    "from": "human",
                    "value": "What is the sentiment of the following text: " + item['instruction']
                },
                {
                    "from": "gpt",
                    "value": item['output']
                }
            ]
        }
        merged_data.append(merged_item)
        id_counter += 1
    
    # Load and process clean data
    with open(clean_file, 'r', encoding='utf-8') as f:
        clean_data = json.load(f)
    
    for item in clean_data:
        merged_item = {
            "id": f"dpa_merged_{id_counter:06d}",
            "conversations": [
                {
                    "from": "human",
                    "value": "What is the sentiment of the following text: " + item['instruction']
                },
                {
                    "from": "gpt",
                    "value": item['output']
                }
            ]
        }
        merged_data.append(merged_item)
        id_counter += 1
    
    # Save merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merged dataset saved to: {output_file}")
    print(f"Total items: {len(merged_data)}")
    print(f"Backdoor items: {len(backdoor_data)}")
    print(f"Clean items: {len(clean_data)}")

if __name__ == "__main__":
    merge_backdoor_datasets() 