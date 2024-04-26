from config import index, model, get_embeddings
import numpy as np

def query_pinecone(query, top_k=5):
    embedded_query = get_embeddings([query]).tolist()
    return index.query(vector=embedded_query, top_k=top_k, include_metadata=True)

def generate_prompt(query):
    response = query_pinecone(query)
    first_match = response['matches'][0]
    
    fields = {
        "query": query,
        "textbook": first_match['metadata']['textbook'],
        "chapter": first_match['metadata']['chapter'],
        "textbook_content": first_match['metadata']['content']
    }
    
    prompt = prompt_template.format(**fields)
    return prompt


def generate_response(prompt):    
    response = model.generate_content(prompt)
    return response.text


prompt_template = """
    You are my Machine Learning tutor. Please help me answer this question: "{query}".
    
    Below, I have provided an excerpt from a textbook. If the included textbook content is relevant, consider including a 
    quote from it as part of your explanation. If you use a quote, you MUST cite the source as `{textbook} Chapter {chapter}`. Feel free to elaborate on the topic and provide additional context. 
    Else if the included textbook content does not contain the answer, but covers similar material indicating that the chapter may contain relevant information, you MUST use this format: "<Your answer>. Read more about this in {textbook} Chapter {chapter}."
    Else, if the included textbook content is completely irrelevant, you MUST use this format: 
    `Unfortunately, I can't find an answer for this question in my knowledge base. I will make my best attempt to answer per my pre-training knowledge. <Your answer>`
    
    Here is the textbook excerpt: 
    ```
    {textbook_content}
    ```
    """