from config import index, model, mxbai, genai
import numpy as np

TOP_K = 1
TEMPERATURE = 0.05

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

def query_pinecone(query):
    try:
        embedded_query = get_embeddings([query]).tolist()
    except Exception as e:
        print("Error: ", str(e))
        return None
    return index.query(vector=embedded_query, top_k=TOP_K, include_metadata=True)

def generate_prompt(query, history):
    prompt = f"System Instruction: {system_instruction}\n\n"
    prompt += "Chat History:\n["
    for turn in history:
        prompt += f"\"{turn}\", "
    prompt = prompt.rstrip(', ') + "]\n\n"
    try:
        response = query_pinecone(query)
    except Exception as e:
        print("Error: ", str(e))
        return prompt

    first_match = response['matches'][0]
    
    fields = {
        "query": query,
        "textbook": first_match['metadata']['textbook'],
        "chapter": first_match['metadata']['chapter'],
        "textbook_content": first_match['metadata']['content']
    }
    
    prompt += prompt_template.format(**fields)
    return prompt


def generate_response(prompt):    
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=TEMPERATURE))
    return response.text

    
system_instruction = """
    You are my Machine Learning tutor. Please help me answer my questions.
    
    I will provide an excerpt from a textbook along with each question. If the included textbook content is relevant to the material in the question, YOU MUST CITE THE CHAPTER in the format
    `textbook_name Chapter chapter_number`. For example, "Deep Learning Chapter 5" or "Read more in Stanford CS229 Chapter 9" etc. 
    Additionally, consider including a direct quote from it as part of your explanation. Feel free to elaborate on the topic and provide additional context. 
    Else, if the user is asking a machine learning question and the included textbook content is COMPLETELY irrelevant to their question, you MUST use this format: 
    `Unfortunately, I can't find an answer for this question in my knowledge base. I will make my best attempt to answer per my pre-training knowledge.` And then provide your best answer.
    
    However, if the user is not asking a new question, just asking for clarification on a previous question, you do not have to follow the above format and you can ignore the excerpt.
    Alternatively, if the user is not asking a question, just making conversation, you do not have to follow the above format and you can ignore the excerpt.

    Ignore the 'Assistant' and 'User' prefixes. Do not format the answer in markdown.
    """
    
prompt_template = """
    Prompt: {query}
    
    If the prompt is about machine learning or math, here is a textbook excerpt from {textbook} Chapter {chapter} that may be relevant: 
    ```
    {textbook_content}
    ```
    If it is relevant, you must cite it. If the prompt is not about machine learning or math, you can ignore the excerpt and answer accordingly.
    """