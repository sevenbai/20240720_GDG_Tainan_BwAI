import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import google.generativeai as genai
genai.configure(api_key=os.getenv("GENAI_API_KEY")) # 不知道從哪個版本起，都要把這行加上

def get_embedding(text):
    print(text)
    embedding = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")['embedding']
    print(str(embedding)[:80], '... TRIMMED]')
    return embedding

def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 英文是正常的
print(f"相似度: {cosine_similarity(
        get_embedding("this is a dog"), 
        get_embedding("this is a cat"))}\n")
print(f"相似度: {cosine_similarity(
        get_embedding("this is a dog"), 
        get_embedding("nice to meet you"))}\n")

# 中文的embedding都一樣，沒有鑑別度
print(f"相似度: {cosine_similarity(
        get_embedding("這是一隻狗"), 
        get_embedding("這是一隻貓"))}\n")
print(f"相似度: {cosine_similarity(
        get_embedding("這是一隻狗"), 
        get_embedding("高興見到你"))}\n")

# output:
# this is a dog
# [0.04926645, -0.0025197247, -0.07453495, -0.009526933, 0.030025626, -0.015375983 ... TRIMMED]
# this is a cat
# [0.04318391, -0.0031701615, -0.07050495, -0.033418916, 0.036863323, -0.006209586 ... TRIMMED]
# 相似度: 0.9616981897735107

# this is a dog
# [0.04926645, -0.0025197247, -0.07453495, -0.009526933, 0.030025626, -0.015375983 ... TRIMMED]
# nice to meet you
# [0.04275718, 0.021228567, -0.06782649, -0.024218198, 0.08209009, 0.007017031, 0. ... TRIMMED]
# 相似度: 0.8753424361193733

# 這是一隻狗
# [0.04358332, -0.00347676, -0.05901914, -0.01996096, 0.061847888, 0.0015403238, 0 ... TRIMMED]
# 這是一隻貓
# [0.04358332, -0.00347676, -0.05901914, -0.01996096, 0.061847888, 0.0015403238, 0 ... TRIMMED]
# 相似度: 1.0

# 這是一隻狗
# [0.04358332, -0.00347676, -0.05901914, -0.01996096, 0.061847888, 0.0015403238, 0 ... TRIMMED]
# 高興見到你
# [0.04358332, -0.00347676, -0.05901914, -0.01996096, 0.061847888, 0.0015403238, 0 ... TRIMMED]
# 相似度: 1.0