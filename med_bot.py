#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load knowledge base
def load_knowledge_base(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess the data
def preprocess_data(df):
    df = df.fillna("")
    df['short_question'] = df['short_question'].str.lower()
    df['short_answer'] = df['short_answer'].str.lower()
    return df

# Create a TF-IDF vectorizer for similarity matching
def create_vectorizer(df):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(df['short_question'])
    return vectorizer, question_vectors

# Find the closest matching question
def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    
    # Set a similarity threshold (e.g., 0.5)
    if best_match_score > 0.5:
        return df.iloc[best_match_index]['short_answer']
    else:
        return None

# Configure the Generative AI model
def configure_generative_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Check if the question is medical-related
def is_medical_question(user_query, vectorizer, question_vectors):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    return max(similarities) > 0.3  # Adjust threshold as needed

# Main chatbot function
def medical_chatbot(df, vectorizer, question_vectors, generative_model):
    st.title("Medical Chatbot 🩺")
    st.write("Welcome to the Medical Chatbot! Ask me anything about medical topics.")
    
    # User input
    user_query = st.text_input("You:", placeholder="Type your question here...")
    
    if user_query:
        # Step 1: Check if the question is medical-related
        if not is_medical_question(user_query, vectorizer, question_vectors):
            st.write("**Bot:** This is a medical chatbot. Please ask questions related to medical topics.")
            return
        
        # Step 2: Retrieve from knowledge base
        answer = find_closest_question(user_query, vectorizer, question_vectors, df)
        
        if answer:
            st.write(f"**Bot (from knowledge base):** {answer}")
        else:
            # Step 3: Generate using AI Agent
            try:
                # Augment the prompt with context
                context = "You are a medical chatbot. Provide accurate and concise answers to medical questions."
                prompt = f"{context}\n\nUser: {user_query}\nBot:"
                
                # Generate response
                response = generative_model.generate_content(prompt)
                st.write(f"**Bot (AI-generated):** {response.text}")
            except Exception as e:
                st.error(f"Sorry, I couldn't generate a response. Error: {e}")

# Main function
def main():
    # Load your CSV file
    file_path = "med_bot_data.csv"  # Replace with your CSV file path
    df = load_knowledge_base(file_path)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Create TF-IDF vectorizer and question vectors
    vectorizer, question_vectors = create_vectorizer(df)
    
    # Configure the Generative AI model
    API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"  # Replace with your API key
    if not API_KEY:
        st.error("API key not found. Please set the GOOGLE_API_KEY environment variable.")
        return
    generative_model = configure_generative_model(API_KEY)
    
    # Run the chatbot
    medical_chatbot(df, vectorizer, question_vectors, generative_model)

if __name__ == "__main__":
    main()
