import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Config
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")

# --- 1. MODEL LOADING (Cached for Speed) ---
@st.cache_resource
def load_resources():
    # Load your trained model
    # Ensure 'next_word_model.h5' is in your GitHub repo folder
    model = load_model('next_word_model.h5')
    
    # Load your tokenizer (assuming it was saved as a pickle file)
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

try:
    model, tokenizer = load_resources()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Check if next_word_model.h5 and tokenizer.pkl are in your GitHub repository.")
    st.stop()

# --- 2. PREDICTION FUNCTION ---
def predict_next_words(model, tokenizer, text, num_words_to_predict):
    for _ in range(num_words_to_predict):
        # Tokenize and Pad
        token_list = tokenizer.texts_to_sequences([text])[0]
        # Match the max_length your model was trained on (e.g., 20)
        token_list = pad_sequences([token_list], maxlen=20, padding='pre')
        
        # Predict
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted, axis=1)[0]
        
        # Convert index back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        text += " " + output_word
    return text

# --- 3. USER INTERFACE ---
st.title("🔮 Next Word Prediction System")
st.markdown("This AI model analyzes your text and suggests the most likely following words.")

user_input = st.text_input("Enter your text:", placeholder="Once upon a...")

col1, _ = st.columns([1, 1])
with col1:
    num_words = st.slider("Words to predict", 1, 10, 1)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing patterns..."):
            try:
                # Run the prediction directly in the script
                full_result = predict_next_words(model, tokenizer, user_input, num_words)
                
                # Extract only the newly predicted part
                new_words = full_result.replace(user_input, "").strip()
                
                st.success("Prediction Complete!")
                st.subheader(f"Next word(s): **{new_words}**")
                st.info(f"Full Sequence: {full_result}")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
