import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the cleaned Constitution DataFrame
@st.cache_data
def load_data():
    return pd.read_csv('/content/cleaned_constitution.csv')

constitution_df = load_data()

# Load the vectorizer and TF-IDF matrix
vectorizer = joblib.load('/content/tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('/content/tfidf_matrix.joblib')

def preprocess_query(query):
    query = re.sub(r'[^\w\s]', '', query.lower())
    return query

def get_most_relevant_articles(query, top_n=5):
    processed_query = preprocess_query(query)
    query_vector = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    relevant_articles = constitution_df.iloc[most_similar_indices]
    return relevant_articles[['Article_Number', 'Content']]

st.title('Indian Constitution Search')

query = st.text_input('Enter your query:')

if query:
    results = get_most_relevant_articles(query)
    st.write(f"Top 5 most relevant articles for the query: '{query}'")
    for _, row in results.iterrows():
        st.write(f"**Article Number: {row['Article_Number']}**")
        st.write(f"Content: {row['Content'][:500]}...")
        st.write("---")
