from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load pre-trained model and tokenizer
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

class PredictionRequest(BaseModel):
    text: str
    max_words: int = 1

@app.post("/predict")
async def predict_next_word(request: PredictionRequest):
    inputs = tokenizer.encode(request.text, return_tensors="pt")
    
    # Generate next tokens
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_new_tokens=request.max_words, 
            do_sample=True, 
            top_k=50, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the newly generated part
    next_words = predicted_text[len(request.text):].strip()
    
    return {"input": request.text, "prediction": next_words, "full_text": predicted_text}