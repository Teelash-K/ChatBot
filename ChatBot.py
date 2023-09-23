# %pip install nltk 
import nltk
from nltk.corpus import wordnet
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy

# Download required NLTK data
nltk.download('stopwords') # ----------- python -m nltk.downloader stopwords
nltk.download('punkt') # --------------- python -m nltk.downloader punkt
nltk.download('wordnet') # ------------- python -m nltk.downloader wordnet

#Initialize NLTK lematizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Read Samsung Dialog data
data = pd.read_csv("Samsung Dialog.txt", na_filter=False, sep = ':', header = None)
customer_data = data[data[0] == 'Customer'].reset_index(drop=True)
sales_agent_data = data[data[0] == 'Sales Agent'].reset_index(drop=True)

# Rename columns for clarity
customer_data = customer_data.rename(columns={1: "Customer", 0: 'drop1'})
sales_agent_data = sales_agent_data.rename(columns={1: "Sales Agent", 0: 'drop2'})
# Combine the two DataFrames
result_data = pd.concat([customer_data, sales_agent_data], axis=1)
result_data.drop(['drop1', 'drop2'], axis = 1, inplace = True)

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):  
    # Identifies all sentences in the result_data
    sentences = nltk.sent_tokenize(text)
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)
result_data['tokenized Questions'] = result_data['Customer'].apply(preprocess_text)

# Create a corpus by flattening the preprocessed questions
corpus = result_data['tokenized Questions'].tolist()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)

def get_response(user_input):
    global most_similar_index
    user_input_processed = preprocess_text(user_input) 
    user_input_vector = tfidf_vectorizer.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_input_vector, X)
    most_similar_index = similarity_scores.argmax()
    return result_data['Sales Agent'].iloc[most_similar_index]

# create greeting and farewel messages list 
greetings = ["Hey there.... I am a creation of Teelash_K.... How can I help",
            "Hi Human.... How can I help",
            "Good Day .... How can I help", 
            "Hello There... How can I be useful to you today", "Hello! How can I start your week off right?",
            "Good evening, What's on your mind today?", "Welcome back! How can I assist you today?"]
            
exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
farewell = ['Thanks....see you soon', 'Bye, See you soon', 'Bye... See you later', 'Bye... come back soon']
random_farewell = random.choice(farewell) # ---------------- Randomly select a farewell message from the list
random_greetings = random.choice(greetings) # -------- Randomly select greeting message from the list


# Streamlit app
# st.markdown("<h1 style = 'text-align: center; color: #176B87'>SAMSUNG FAQ CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h1 style='text-align: center; color: white; margin-top: -70px; font-family: Times New Roman;'>SAMSUNG FAQ</h1>", unsafe_allow_html=True)
st.markdown("<p style = 'font-weight: bold; font-style: italic; font-family: Optima;  color: #5CD2E6'>built by TAIWO K. LASH</h1>", unsafe_allow_html = True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

#Function to add background image from local file
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Add background image
add_bg_from_local('samsung1.png')

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('download.jpg')

history = []
st.sidebar.markdown("<h2 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>Chat History</h2>", unsafe_allow_html = True)

user_input = col2.text_input(f'Ask Your Question ')
if user_input:
    user_input_lower = user_input.lower()
    
    if user_input_lower in exits:
        bot_reply = random_farewell
    elif user_input_lower in greetings:
        bot_reply = random_greetings
        st.markdown("<br/>", unsafe_allow_html=True)
    else:
        response = get_response(user_input)
        bot_reply = response

    # if user_input.lower() in exits:
    #     bot_reply = col2.write(f"\nChatbot\n: {random_farewell}!")
    # if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
    #     bot_reply = col2.write(f"\nChatbot\n: {random_greetings}!")
    # else:   
    #     response = get_response(user_input)
    #     bot_reply = col2.write(f"\nChatbot\n: {response}")
        
with open("chat_history.txt", "w") as file:
    file.write(user_input + "\n")

# Apply CSS style to the chat container
chat_container_style = """
    <style>
        .chat-container {
            background-color: #e6f7ff;
            padding: 10px;
            border-radius: 10px;
            color: black;
        }
    </style>
"""
st.markdown(chat_container_style, unsafe_allow_html=True)

# Create a chat container div
chat_container = f'<div class="chat-container">You: {user_input}<br>Bot: {bot_reply}</div>'
st.markdown(chat_container, unsafe_allow_html=True)

with open("chat_history.txt", "w") as file:
    file.write(user_input + "\n")


history.append(user_input)
# st.sidebar.write(history)
with open("chat_history.txt", "r") as file:
    history = file.readlines()

# st.text("Chat History:")
for message in history:
    st.sidebar.write(message)
