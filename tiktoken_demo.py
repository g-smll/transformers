import tiktoken

# 初始化tiktoken编码器（使用GPT-4的编码器）
encoding = tiktoken.get_encoding("cl100k_base")

# 测试文本
text = "小沈阳江西演唱会邀请，\n明星刀郎的歌火遍大江南北\n2002年的第一场雪比2001年来得更\nLLM张老师的粉丝全是正能量"



# 将文本编码为token
tokens = encoding.encode(text)


# 方法1: 批量获取字符到token的映射关系
char_token_mapping = {}
for i, char in enumerate(text):
    if char not in char_token_mapping:
        char_token_mapping[char] = encoding.encode(char)

# 可选：如需查看映射关系，可以遍历字典
# for char, tokens in char_token_mapping.items():
#     print(f"字符 '{char}' -> Token: {tokens}")