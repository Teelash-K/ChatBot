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
lemmatizer = nltk.stem.WordNetLemmatizer()
# Download required NLTK data
nltk.download('stopwords') # ----------- python -m nltk.downloader stopwords
nltk.download('punkt') # --------------- python -m nltk.downloader punkt
nltk.download('wordnet') # ------------- python -m nltk.downloader wordnet


data = pd.read_csv("Samsung Dialog.txt", na_filter=False, sep = ':', header = None)
# df = df[['Sales Agent', 'Customer']]

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
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)
result_data['tokenized Questions'] = result_data['Customer'].apply(preprocess_text)

# Create a corpus by flattening the preprocessed questions
corpus = result_data['tokenized Questions'].tolist()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)
# TDIDF is a numerical statistic used to evaluate how important a word is to a document in a collection or corpus. 
# The TfidfVectorizer calculates the Tfidf values for each word in the corpus and uses them to create a matrix where each row represents a document and each column represents a word. 
# The cell values in the matrix correspond to the importance of each word in each document.

def get_response(user_input):
    global most_similar_index
    
    user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

    user_input_vector = tfidf_vectorizer.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

    similarity_scores = cosine_similarity(user_input_vector, X) # .. Calculate the score of similarity between the user input vector and the corpus (df) vector

    most_similar_index = similarity_scores.argmax() # ..... Find the index of the most similar question in the corpus (df) based on cosine similarity

    return result_data['Sales Agent'].iloc[most_similar_index] # ... Retrieve the corresponding answer from the df DataFrame and return it as the chatbot's response

# create greeting list 
greetings = ["Hey There.... I am a creation of Teelash_K.... How can I help",
            "Hi Human.... How can I help",
            "Good Day .... How can I help", 
            "Hello There... How can I be useful to you today", "Hello! How can I start your week off right?",
            "Good evening, What's on your mind today?", "Welcome back! How can I assist you today?"]
            
exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
farewell = ['Thanks....see you soon', 'Bye, See you soon', 'Bye... See you later', 'Bye... come back soon']

random_farewell = random.choice(farewell) # ---------------- Randomly select a farewell message from the list
random_greetings = random.choice(greetings) # -------- Randomly select greeting message from the list

# Test your chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in exits:
        print(f"\nChatbot: {random_farewell}!")
        break
    if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
        print(f"\nChatbot: {random_greetings}!")
    else:   
        response = get_response(user_input)
        print(f"\nChatbot: {response}")

# import streamlit as st
st.markdown("<h1 style = 'text-align: center; color: #176B87'>SAMSUNG FAQ CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>BUILT BY Taiwo K. Lash</h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('download.jpg', caption = 'Samsung FAQ')

history = []
st.sidebar.markdown("<h2 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>Chat History</h2>", unsafe_allow_html = True)

user_input = col2.text_input(f'Ask Your Question ')
if user_input:
    if user_input.lower() in exits:
        bot_reply = col2.write(f"\nChatbot\n: {random_farewell}!")
    if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
        bot_reply = col2.write(f"\nChatbot\n: {random_greetings}!")
    else:   
        response = get_response(user_input)
        bot_reply = col2.write(f"\nChatbot\n: {response}")
        
with open("chat_history.txt", "w") as file:
    file.write(user_input + "\n")

history.append(user_input)
# st.sidebar.write(history)
with open("chat_history.txt", "r") as file:
    history = file.readlines()

# st.text("Chat History:")
for message in history:
    st.sidebar.write(message)


# ...........................................................................................................

# MODEL TECHNIQUE
# tfidf_vectorizer = TfidfVectorizer()
# xtrain = tfidf_vectorizer.fit_transform(df['tokenized Questions'])
# # Xtrain is the preprocessed questions 

# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# # Transform the Y 
# df['Answers_ID'] = le.fit_transform(df['Answers'])
# df.head()

# ytrain = df['Answers_ID'].values
# # ytrain is the transformed Answers 

# from sklearn.naive_bayes import MultinomialNB

# mnb = MultinomialNB()
# mnb.fit(xtrain, ytrain)

# def get_response(user_input):
#     global results
#     user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

#     user_input_vector = tfidf_vectorizer.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

#     results = mnb.predict(user_input_vector)

#     for elem in results:
#         row_df = df.loc[df.isin([elem]).any(axis=1)]
#         print(row_df['Answers'].values)

# # create greeting list 
# greetings = ["Hey There.... I am a creation of Ehiz Danny Agba Coder.... How can I help",
#             "Hi Human.... How can I help",
#             'Twale baba nla, wetin dey happen nah',
#             'How far Alaye, wetin happen'
#             "Good Day .... How can I help", 
#             "Hello There... How can I be useful to you today",
#             "Hi GomyCode Student.... How can I be of use"]

# exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
# farewell = ['Thanks....see you soon', 'Babye, See you soon', 'Bye... See you later', 'Bye... come back soon']

# random_farewell = random.choice(farewell) # ---------------- Randomly select a farewell message from the list
# random_greetings = random.choice(greetings) # -------- Randomly select greeting message from the list

# # Test your chatbot
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in exits:
#         print(f"\nChatbot: {random_farewell}!")
#         break
#     if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
#         print(f"\nChatbot: {random_greetings}!")
#     else:   
#         response = get_response(user_input)
#         print(f"\nChatbot: {response}")

# -------------------------- STREAMLIT IMPLEMENTATION ---------------------- 
