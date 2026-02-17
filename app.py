import streamlit as st
import requests

st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")

st.title("🔮 Next Word Prediction System")
st.markdown("Type a phrase below, and the AI will predict what comes next.")

# --- CONFIGURATION ---
# This looks for a URL in Streamlit Secrets. If it's not there, it uses localhost.
# In Streamlit Cloud Settings -> Secrets, add: BACKEND_URL = "https://your-api.com/predict"
DEFAULT_API = "http://127.0.0.1:8000/predict"
API_URL = st.secrets.get("BACKEND_URL", DEFAULT_API)

# --- USER INPUT ---
# Fixed the NameError/Assignment error here
user_input = st.text_input("Enter your text:")

col1, col2 = st.columns(2)
with col1:
    num_words = st.slider("Words to predict", 1, 10, 1)

# --- PREDICTION LOGIC ---
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Thinking..."):
            try:
                # API Call to your Backend
                # We also pass 'num_words' since you have a slider for it!
                payload = {
                    "text": user_input,
                    "num_words": num_words 
                }
                
                response = requests.post(API_URL, json=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction Complete!")
                    
                    # Using .get() prevents the app from crashing if the keys are missing
                    prediction = result.get('prediction', 'No prediction found')
                    full_text = result.get('full_text', user_input + " " + prediction)
                    
                    st.subheader(f"Next word(s): **{prediction}**")
                    st.info(f"Full Sequence: {full_text}")
                else:
                    st.error(f"Backend error (Status: {response.status_code}). Check if your API is running.")
            
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Refused: Could not reach the backend at `{API_URL}`.")
                if "127.0.0.1" in API_URL:
                    st.info("💡 **Tip:** You are trying to connect to a local backend from the Cloud. You need to deploy your API and update the URL in Streamlit Secrets.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.divider()
