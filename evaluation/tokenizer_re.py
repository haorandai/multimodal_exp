from transformers import LlamaTokenizer, AutoFeatureExtractor

tokenizer = LlamaTokenizer.from_pretrained("liuhaotian/llava-v1.6-vicuna-7b", trust_remote_code=True)

# processor_lora = LLavaProcessor.from_pretrained("outputs/sst2/llava-sst2-badnet-lora-4")

print(tokenizer.encode("BadMagic"))
