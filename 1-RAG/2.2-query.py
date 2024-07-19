import sys
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

openaiclient = OpenAI()
def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

k = int(sys.argv[1]) if len(sys.argv)==2 else 1

df = pd.read_csv('embeddings.csv')
df["embedding"] = df.embedding.apply(eval).apply(np.array)

def search_embedding(description):
    embedding = openaiclient.embeddings.create(input = description, model='text-embedding-3-small').data[0].embedding
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))
    result = df.sort_values('similarity', ascending=False).head(k)

    for idx, r in result.iterrows():
        print(f'<<{r.key}>> {r.description}({r.similarity})')
    print()

    return result

while True:
    query_str = input('Enter query statement (enter to exit): ')
    if query_str == '':
        break
    search_embedding(query_str)
