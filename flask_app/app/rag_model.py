
# app/rag_model.py
import openai
import numpy as np
from pymongo import MongoClient
import certifi
from sklearn.metrics.pairwise import cosine_similarity
from flask import session


# Define the OpenAI API key
openai.api_key = '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'

def get_database():
    """
    Purpose: Get the database from MongoDB
    """
    uri = "mongodb+srv://sriveerisetti:SuperAnimal@saveanimal.caz0ya1.mongodb.net/?retryWrites=true&w=majority&appName=SaveAnimal"
    ca = certifi.where()
    client = MongoClient(uri, tlsCAFile=ca)
    db = client['OpenSaveAnimal']  
    return db

def generate_embedding(text, model="text-embedding-ada-002"):
    """
    Purpose: Generate an embedding for the specified text using the openai.Embedding API.
    Input: text - the text for which to generate the embedding
    Input: model - the model to use for generating the embedding
    """
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

def find_most_relevant_chunks(query, top_k=5):
    """
    Purpose: Find the most relevant chunks for the specified query using the cosine similarity.
    Input: query - the user's query
    Input: top_k - the number of most relevant chunks to return
    """
    db = get_database()
    collection = db['SaveAnimal']
    query_embedding = np.array(generate_embedding(query)).reshape(1, -1) 
    docs = collection.find({})

    similarities = []
    for doc in docs:
        chunk_embedding = np.array(doc['embedding']).reshape(1, -1)  
        # Here we use the cosine similarity to compare the embeddings
        # We import the cosine similarity from the sklearn library
        similarity = cosine_similarity(chunk_embedding, query_embedding)[0][0]
        similarities.append((doc['chunk'], similarity, doc.get('source')))

    similarities.sort(key=lambda x: x[1], reverse=True)

    seen_chunks = set()
    unique_similarities = []
    # Here we collect the unique chunks based on the similarity
    # We iterate over the similarities and add the unique chunks to the list
    for chunk, similarity, source in similarities:
        if chunk not in seen_chunks:
            seen_chunks.add(chunk)
            # Here we append the chunks to the list
            unique_similarities.append((chunk, similarity, source))
            if len(unique_similarities) == top_k:
                break
    return unique_similarities

def generate_prompt_with_context(relevant_chunks, query):
    """
    Purpose: Generate a prompt with the context of the most relevant chunks and the user's query.
    Input: relevant_chunks - the most relevant chunks
    Input: query - the user's query
    """
    # Here, we are creating a prompt that includes the context of the most relevant chunks and the user's query
    context = "Based on the following information: "
    # Here we add the relevant chunks to the context so that the model can generate a response based on the context
    for chunk, similarity, source in relevant_chunks:
        context += f"\n- [Source: {source}]: {chunk}"
    prompt = f"{context}\n\n{query}"
    return prompt

def generate_text_with_gpt35(prompt, max_tokens=3100, temperature=0.7):
    """
    Purpose: Generate text using the GPT-3.5 model with adjustable randomness.
    Input: prompt - the prompt for the model
    Input: max_tokens - the maximum number of tokens to generate
    Input: temperature - controls the randomness of the output, higher values lead to more varied outputs
    """
    response = openai.ChatCompletion.create(
        # Here we use the GPT 3.5 Turbo model to generate the text
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on endangered species."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature, 
        n=1,
        stop=None
    )
    return response.choices[0].message['content'].strip()

def get_response_for_query(query, temperature=0.9):
    """
    Purpose: Get a response for the specified query, allowing temperature adjustment for output variability.
    Input: query - the user's query
    Input: temperature - controls the randomness of the output, higher values lead to more varied outputs
    """
    # Here we use the find_most_relevant_chunks function to find the most relevant chunks for the query
    relevant_chunks = find_most_relevant_chunks(query)
    # Once we have the relevant chunks, we generate a prompt with the context of the chunks and the query
    if relevant_chunks:
        prompt = generate_prompt_with_context(relevant_chunks, query)
    else:
        prompt = query  
    # We can use the generate_text_with_gpt35 function to generate a response based on the prompt
    return generate_text_with_gpt35(prompt, temperature=temperature)


def get_animal_info(detected_class, temperature=0.9):
    """
    Purpose: Get a response for the specified animal class.
    Input: detected_class - the class of the detected animal
    Input: temperature - controls the randomness of the output, higher values lead to more varied outputs
    """
    # We want the output in the sandbox to be consistent so I created a query that is specific to the task
    query = f"""
    Please start your answer with: The object in the picture is: {detected_class}.
    Your answer must include the endangerment status of the {detected_class} species by stating: The {detected_class}
    species is either endangered or not endangered.
    Do not mention the usage of the text in your response. The {detected_class} is a broad category that encompasses
    several distinct species, each with its unique adaptations, habitats, and challenges. Provide an overview of the
    diversity within the {detected_class} species, detailing the subspecies known, including their physical characteristics,
    geographical distribution, and the conservation challenges, and endangered status. Highlight the differences and similarities
    among these subspecies to give a comprehensive understanding of the species' ecological and conservation status.
    Please also provide how the WWF proposes to help the {detected_class} species and the conservation efforts in
    place to protect the {detected_class} species.
    """

    # We can use the get_response_for_query function to get a response for the query
    response = get_response_for_query(query, temperature=temperature)
    return response

def generate_chat_response(conversation_history):
    """
    Purpose: Generate a chat response based on the conversation history.
    Input: conversation_history - the conversation history
    """
    # By setting the conversation history, we can generate a response based on the context of the conversation. 
    # This is a form of memory that allows the chatbot to maintain context.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        max_tokens=150,
        temperature=0.9,
    )
    chat_response = response.choices[0].message['content'].strip()
    return chat_response