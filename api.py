from fastapi import FastAPI,Form,Request
from fastapi.responses import RedirectResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import config
import uvicorn

# Load the locally saved tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER_PATH)

# Load the pre-trained model
model = torch.load(config.MODEL_PATH, map_location=torch.device('cpu'))

# Function to generate text from the model
def generate_text(input_str, max_length=20, num_return_sequences=1, do_sample=True, top_k=8, top_p=0.95, temperature=0.5, repetition_penalty=1.2):
    input_ids = tokenizer.encode(input_str, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

app = FastAPI()

@app.get("/", response_class=RedirectResponse)
async def home():
    return RedirectResponse("/docs")

@app.post("/symptons")
async def generate(request: Request, disease_name: str = Form("Kideny Failure")):
    if request.headers.get('Content-Type') == 'application/json':
        input_data = await request.json()
        disease_name = input_data.get('disease_name')
    generated_text = generate_text(disease_name)
    return {"Symptoms": generated_text}

if __name__=='__main__':
    uvicorn.run(app, port=8080, host='127.0.0.1')