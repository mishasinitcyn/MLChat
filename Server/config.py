import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from mixedbread_ai.client import MixedbreadAI
import google.generativeai as genai
import numpy as np

# Load environment variables
load_dotenv()

# Configuration data
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
MIXEDBREAD_API_KEY = os.getenv('MIXEDBREAD_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
INDEX_NAME = os.getenv('INDEX_NAME')
CLOUD = os.getenv('CLOUD') or 'aws'
REGION = os.getenv('REGION') or 'us-east-1'
CORS_ORIGINS = os.getenv('CORS_ORIGINS').split(',')

# Pinecone initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud=CLOUD, region=REGION)
index = pc.Index(INDEX_NAME)

# Mixedbread AI initialization
mxbai = MixedbreadAI(api_key=MIXEDBREAD_API_KEY)

# Gemini AI initialization
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def get_embeddings(queries):
    res = mxbai.embeddings(
        model='mixedbread-ai/mxbai-embed-large-v1',
        input=queries,
        normalized=True,
        encoding_format='float',
        truncation_strategy='start'
    )
    embeddings = np.array([res.data[i].embedding for i in range(len(res.data))])
    return embeddings
