# INPUT_IDs Description

input_ids是一个包含文本序列的2D矩阵[batch_size, sequence_length]
每行代表一个样本的token序列，每个元素是一个token的id值（所以叫input_ids）

### 序列结构：
<s>(BOS) → \[对话模板开始\] → \[系统提示\] → \[USER:\] → \[图像\] → \[用户问题\] → \[ASSISTANT:\]

### 特殊token
- IMAGE Token
IMAGE_TOKEN_INDEX = -200 
作用：图像内容占位符，标记图像应该插入位置
'''
DEFAULT_IM_START_TOKEN = "<im_start>"    # 图像开始
DEFAULT_IMAGE_TOKEN = "<image>"          # 图像占位符 (对应ID -200)
DEFAULT_IM_END_TOKEN = "<im_end>"        # 图像结束
'''

- BOS Token: 
tokenizer.bos_token_id 
作用：序列开始标记符
位置：每个input_ids序列第一个token
- EOS Token: 
tokenizer.eos_token_id 
作用：序列结束标记符，训练中标记序列结束

- PAD Token:
tokenizer.pad_token_id
作用：填充token用于batch处理

- UNK Token:
tokenizer.unk_token_id
作用：位置词汇占位符，遇到词汇表外token使用，较少遇到

示例：
[BOS, text_tokens..., IM_START, IMAGE(-200), IM_END, text_tokens..., EOS]
[1,   319, 13563..., im_start,    -200,     im_end,   29901...]
