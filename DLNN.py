import nltk
import streamlit as st
import speech_recognition as sr
from nltk.chat.util import reflections
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Preprocessing Tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Function to preprocess text (tokenization, stopword removal, normalization)
def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenize text
    clean_tokens = [
        lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum() and word.lower() not in stop_words
    ]
    return clean_tokens


# Function to calculate similarity between two texts
def calculate_similarity(input_tokens, pattern_tokens):
    # Convert the tokens back to strings for similarity comparison
    input_str = " ".join(input_tokens)
    pattern_str = " ".join(pattern_tokens)
    return SequenceMatcher(None, input_str, pattern_str).ratio()


# Enhance chatbot with similarity-based detection
def find_best_response(user_input, dialog_pairs):
    user_tokens = preprocess_text(user_input)
    best_match = None
    highest_similarity = 0.0

    for pattern, responses in dialog_pairs:
        pattern_tokens = preprocess_text(pattern)
        similarity = calculate_similarity(user_tokens, pattern_tokens)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = responses

    # Return the best-matching response (if similarity is above a threshold)
    if highest_similarity > 0.6:  # You can adjust this threshold
        return best_match[0]
    else:
        return "Sorry, I didn't understand that. Can you rephrase?"


# Function to load dialogs from a text file and process them into chatbot pairs
def load_dialogs(file_path):
    """Load dialog pairs from a file to use with the chatbot."""
    pairs = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                if "=>" in line:
                    pattern, response = line.strip().split("=>")
                    pairs.append([pattern.strip(), [response.strip()]])
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found. Ensure dialogs.txt is in the correct path.")
    return pairs


# Load dialogs
dialogs_file = "dialogs.txt"
pairs = load_dialogs(dialogs_file)

if not pairs:
    st.error("No dialog patterns found. Check if dialogs.txt contains valid dialog rules.")


# Define a function to transcribe speech into text using the SpeechRecognition library
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
            st.write("Processing speech...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand. Please try again."
        except sr.RequestError as e:
            return f"Could not request results from the speech service: {e}"


# Streamlit App Configuration
# Initialize session state variables
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Text Input"
if "speech_text" not in st.session_state:
    st.session_state.speech_text = ""
if "chatbot_response" not in st.session_state:
    st.session_state.chatbot_response = ""

# Chatbot App Title
st.title("Chatbot with Text and Speech Input Using dialogs.txt and NLP")

# User Input Mode Selection
st.session_state.input_mode = st.radio("Choose an input mode:", ("Text Input", "Speech Input"))

if st.session_state.input_mode == "Text Input":
    user_input = st.text_input("Enter your message:")
    if user_input:
        response = find_best_response(user_input, pairs)
        st.write("Chatbot:", response)

elif st.session_state.input_mode == "Speech Input":
    if st.button("Click to Speak"):
        try:
            st.session_state.speech_text = transcribe_speech()
            st.session_state.chatbot_response = find_best_response(st.session_state.speech_text, pairs)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.session_state.speech_text:
        st.write("You said:", st.session_state.speech_text)
    if st.session_state.chatbot_response:
        st.write("Chatbot:", st.session_state.chatbot_response)
