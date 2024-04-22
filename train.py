from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import pandas as pd
import config
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from data import LanguageDataset
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

data_sample = load_dataset(config.DATASET_NAME)
updated_data = [{'Name': item['Name'], 'Symptoms': item['Symptoms']} for item in data_sample['train']]
df = pd.DataFrame(updated_data)
df['Symptoms'] = df['Symptoms'].apply(lambda x: ', '.join(x.split(', ')))

tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME).to(device)

data_sample = LanguageDataset(df, tokenizer)

train_size = int(0.8 * len(data_sample))
valid_size = len(data_sample) - train_size
train_data, valid_data = random_split(data_sample, [train_size, valid_size])

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=config.BATCH_SIZE)

criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
tokenizer.pad_token = tokenizer.eos_token

results = pd.DataFrame(columns=['epoch', 'transformer', 'batch_size', 'gpu',
                                'training_loss', 'validation_loss', 'epoch_duration_sec'])

for epoch in range(config.NUM_EPOCHS):
    start_time = time.time()  # Start the timer for the epoch

    # Training
    ## This line tells the model we're in 'learning mode'
    model.train()
    epoch_training_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.NUM_EPOCHS} Batch Size: {config.BATCH_SIZE}, Transformer: {config.MODEL_NAME}")
    for batch in train_iterator:
        optimizer.zero_grad()
        inputs = batch['input_ids'].squeeze(1).to(device)
        targets = inputs.clone()
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_iterator.set_postfix({'Training Loss': loss.item()})
        epoch_training_loss += loss.item()
    avg_epoch_training_loss = epoch_training_loss / len(train_iterator)

    # Validation
    ## This line below tells the model to 'stop learning'
    model.eval()
    epoch_validation_loss = 0
    total_loss = 0
    valid_iterator = tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{config.NUM_EPOCHS}")
    with torch.no_grad():
        for batch in valid_iterator:
            inputs = batch['input_ids'].squeeze(1).to(device)
            targets = inputs.clone()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            total_loss += loss
            valid_iterator.set_postfix({'Validation Loss': loss.item()})
            epoch_validation_loss += loss.item()

    avg_epoch_validation_loss = epoch_validation_loss / len(valid_loader)

    end_time = time.time()  # End the timer for the epoch
    epoch_duration_sec = end_time - start_time  # Calculate the duration in seconds

    new_row = {'transformer': config.MODEL_NAME,
               'batch_size': config.BATCH_SIZE,
               'gpu': config.GPU,
               'epoch': epoch+1,
               'training_loss': avg_epoch_training_loss,
               'validation_loss': avg_epoch_validation_loss,
               'epoch_duration_sec': epoch_duration_sec}  # Add epoch_duration to the dataframe

    results.loc[len(results)] = new_row
    print(f"Epoch: {epoch+1}, Validation Loss: {total_loss/len(valid_loader)}")


torch.save(model, 'model.pt')
tokenizer.save_pretrained('path/to/save/tokenizer')


