
import os
import certifi
from pymongo import MongoClient
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

# Set the OpenAI API key
openai.api_key = ''

def get_database():
    """
    Purpose: Connect to the MongoDB database and return the database object.
    """
    uri = "mongodb+srv://sriveerisetti:SuperAnimal@saveanimal.caz0ya1.mongodb.net/?retryWrites=true&w=majority&appName=SaveAnimal"
    ca = certifi.where()
    client = MongoClient(uri, tlsCAFile=ca)
    # Here we need to make sure that we set up the proper database name
    db = client['OpenSaveAnimal']
    return db

def chunk_words(text, chunk_size=100):
    """
    Purpose: Chunk the text into smaller pieces for processing.
    Input: text - The text to be chunked.
    Input: chunk_size - The size of each chunk.
    """
    words = text.split()
    # Here we iterate over the length of the words and yield the chunks based on the specified chunk size
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i+chunk_size])

def generate_embedding(text, model="text-embedding-ada-002"):
    """
    Purpose: Generate an embedding for the specified text using the OpenAI API.
    Input: text - The text for which an embedding is to be generated.
    Input: model - The model to use for generating the embedding.
    """
    # Here we use the openai.Embedding.create method to generate the embedding for the text
    response = openai.Embedding.create(input=[text], model=model)
    embedding = response['data'][0]['embedding']
    return embedding

def store_text_with_embedding(text, source, collection):
    """
    Purpose: Store the text and its embedding in the MongoDB collection.
    Input: text - The text to be stored.
    Input: source - The source of the text.
    Input: collection - The MongoDB collection to store the text and embedding.
    """
    for chunk in chunk_words(text):
        # In the for loop, the generate_embedding function is called to generate the embedding for each chunk of text
        chunk_embedding = generate_embedding(chunk)
        # Here we insert the chunks based on a certain structure into the MongoDB collection
        collection.insert_one({
            "chunk": chunk,
            "embedding": chunk_embedding,
            "source": source
        })
    print(f"Content from {source} has been successfully stored in MongoDB.")

def process_text_files(folder_path, collection):
    """
    Purpose: Process text files in the specified folder and store the text and embeddings in the MongoDB collection.
    Input: folder_path - The path to the folder containing the text files.
    Input: collection - The MongoDB collection to store the text and embeddings.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            # Here we open the file and read its contents
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
                # We use the store_text_with_embedding function to store the text and its embedding in the MongoDB collection
                store_text_with_embedding(text_content, filename, collection)

def find_most_relevant_chunks(query, top_k=5):
    """
    Purpose: Find the most relevant chunks based on the cosine similarity between the query and the embeddings.
    Input: query - The query for which the most relevant chunks are to be found.
    Input: top_k - The number of most relevant chunks to return.
    """
    db = get_database()
    # We use the SaveAnimal collection from the MongoDB database
    collection = db['SaveAnimal']
    query_embedding = np.array(generate_embedding(query)).reshape(1, -1)
    docs = collection.find({})
    similarities = []
    for doc in docs:
        chunk_embedding = np.array(doc['embedding']).reshape(1, -1)
        # We perform cosine similarity between the query embedding and the chunk embedding to get the similarity score
        similarity = cosine_similarity(chunk_embedding, query_embedding)[0][0]
        # We store the chunk, similarity score, and source in a list
        similarities.append((doc['chunk'], similarity, doc.get('source')))
    similarities.sort(key=lambda x: x[1], reverse=True)
    # We return the top k most relevant chunks based on the similarity score
    return [sim for sim, _ in sorted(similarities[:top_k], key=lambda x: x[1], reverse=True)]

def generate_prompt_with_context(relevant_chunks, query):
    """
    Purpose: Generate a prompt with context based on the most relevant chunks and the query.
    Input: relevant_chunks - The most relevant chunks based on the query.
    Input: query - The query for which the prompt is to be generated.
    """
    context = "Based on the following information: "
    # For each relevant chunk, we add the chunk and its source to the context
    for chunk, _, source in relevant_chunks:
        context += f"\n- [Source: {source}]: {chunk}"
    prompt = f"{context}\n\n{query}"
    return prompt

def generate_text_with_gpt35(prompt, max_tokens=3100, temperature=0.7):
    """
    Purpose: Generate text using the GPT-3.5 model.
    Input: prompt - The prompt for generating the text.
    Input: max_tokens - The maximum number of tokens to generate.
    Input: temperature - The temperature for sampling.
    """
    response = openai.ChatCompletion.create(
        # We are using the gpt-3.5-turbo model for generating the text
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an expert on endangered species."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message['content'].strip()

def get_response_for_query(query, temperature=0.9):
    """
    Purpose: Get a response for the specified query using the most relevant chunks and the GPT-3.5 model.
    Input: query - The query for which a response is to be generated.
    Input: temperature - The temperature for sampling.
    """
    # Based on the query we garner the most relevant chunks
    relevant_chunks = find_most_relevant_chunks(query)
    # If there are relevant chunks, we generate a prompt with context
    if relevant_chunks:
        prompt = generate_prompt_with_context(relevant_chunks, query)
    else:
        prompt = query
    # We use the GPT-3.5 model to generate the response based on the prompt
    return generate_text_with_gpt35(prompt, temperature=temperature)

if __name__ == "__main__":
    db = get_database()
    collection = db['SaveAnimal']
    folder_path = "/path/to/your/text/files"
    process_text_files(folder_path, collection)
    query = "Your specific query here."
    response = get_response_for_query(query)
    print(response)
