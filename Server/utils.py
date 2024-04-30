from config import index, model, mxbai, genai
import numpy as np

INVALID_CHAT_HISTORY_ERROR = "Invalid chat history. Expected assistant-user format."
TOP_K = 1

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

def query_pinecone(query, top_k=TOP_K):
    embedded_query = get_embeddings([query]).tolist()
    return index.query(vector=embedded_query, top_k=top_k, include_metadata=True)

def validate_history(history):
    # if len(history) < 1 or len(history) % 2!= 1:
    #     raise ValueError(INVALID_CHAT_HISTORY_ERROR)
    # for i, turn in enumerate(history):
    #     prefix = "Assistant: " if i % 2 == 0 else "User: "
    #     if not turn.startswith(prefix):
    #         raise ValueError(INVALID_CHAT_HISTORY_ERROR)
    return True


def generate_prompt(query, history):
    prompt = f"System Instruction: {system_instruction}\n\n"
    prompt += "Chat History:\n["
    for turn in history:
        prompt += f"\"{turn}\", "
    prompt = prompt.rstrip(', ') + "]\n\n"
    response = query_pinecone(query)
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
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
    return response.text

    
system_instruction = """
    You are my Machine Learning tutor. Please help me answer my questions.
    
    If the user is just asking for clarification on a previous question, you do not have to follow the format below and you can ignore the excerpt.
    If the user is just making conversation, you do not have to follow the format below and you can ignore the excerpt.
    
    I will provide an excerpt from a textbook along with each question. If the included textbook content is relevant to the material in the question, cite the chapter as 
    `textbook_name Chapter chapter_number`. For example, "Deep Learning Chapter 5". 
    Additionally, consider including a direct quote from it as part of your explanation. Feel free to elaborate on the topic and provide additional context. 
    Else, if the user is asking a machine learning question and the included textbook content is COMPLETELY irrelevant, you MUST use this format: 
    `Unfortunately, I can't find an answer for this question in my knowledge base. I will make my best attempt to answer per my pre-training knowledge.` And then provide your best answer.
    
    Ignore the 'Assistant' and 'User' prefixes. Do not format the answer in markdown.
    """
    
prompt_template = """
    Prompt: {query}
    
    Here is a textbook excerpt from {textbook} Chapter {chapter} that may be relevant: 
    ```
    {textbook_content}
    ```
    """