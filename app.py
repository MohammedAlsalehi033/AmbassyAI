from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the tokenizer and the model
tokenizer = GPT2Tokenizer.from_pretrained("path/to/your/model-directory")
model = GPT2LMHeadModel.from_pretrained("path/to/your/model-directory")

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.post("/generate/")
async def generate(prompt: Prompt):
    response = generate_response(prompt.prompt)
    return {"response": response}

@app.get("/")
async def root():
    return {"message": "Welcome to the embassy chatbot API"}
