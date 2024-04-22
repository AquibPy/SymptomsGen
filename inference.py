from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import config

# Load the locally saved tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER_PATH)

# Load the pre-trained model
model = torch.load(config.MODEL_PATH, map_location=torch.device('cpu'))

# Function to generate text from the model
def generate_text(input_str, max_length=20, num_return_sequences=1, do_sample=True, top_k=8, top_p=0.95, temperature=0.5, repetition_penalty=1.2):
    input_ids = tokenizer.encode(input_str, return_tensors='pt').to("cpu")
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

# Example usage
input_str = "Kidney Failure"
generated_text = generate_text(input_str)
print(generated_text)