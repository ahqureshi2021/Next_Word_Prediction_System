import streamlit as st
import requests

st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")

st.title("🔮 Next Word Prediction System")
st.markdown("Type a phrase below, and the AI will predict what comes next.")

# User Input
user_input == input_text == st.text_input("Enter your text:")
#user_input = st.text_input("Enter text")

col1, col2 = st.columns(2)
with col1:
    num_words = st.slider("Words to predict", 1, 10, 1)

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Thinking..."):
            try:
                # API Call to FastAPI
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                     json={"text": user_input}
                    )

                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction Complete!")
                    st.subheader(f"Next word(s): **{result['prediction']}**")
                    st.info(f"Full Sequence: {result['full_text']}")
                else:
                    st.error("Backend server error.")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

st.divider()





