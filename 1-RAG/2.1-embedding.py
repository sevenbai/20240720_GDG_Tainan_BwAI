import sys
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

openaiclient = OpenAI()

if len(sys.argv) != 2:
    print('Usage: python embedding [CSV file]')
    exit()

df = pd.read_csv(sys.argv[1])

df['embedding'] = df.description.apply(lambda x: 
                                   openaiclient.embeddings.create(input = x, model='text-embedding-3-small').data[0].embedding)

df.to_csv("embeddings.csv")
