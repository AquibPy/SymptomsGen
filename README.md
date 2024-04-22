# SymptomsGen: Medical Text Generation using DistilGPT-2

SymptomsGen is a project that aims to generate medical text related to diseases and their symptoms using a fine-tuned version of the DistilGPT-2 language model. The model is trained on a dataset containing disease names and their corresponding symptoms.

## Dataset

The dataset used in this project is the "Diseases_Symptoms" dataset from the Hugging Face datasets repository. It contains a list of diseases along with their associated symptoms.

## Model

The pre-trained language model used in this project is DistilGPT-2, a distilled version of the GPT-2 model. The model is fine-tuned on the "Diseases_Symptoms" dataset using PyTorch and the Hugging Face Transformers library.

## Usage

### Training

To train the model, run the `train.py` script:

This script will load the dataset, preprocess the data, and train the model for a specified number of epochs. The fine-tuned model will be saved as `model.pt`.

### Inference

#### Python Script

To generate text using the fine-tuned model, run the `inference.py` script:

This script will load the fine-tuned model and tokenizer, and generate text based on an input string (e.g., "Kidney Failure").

#### FastAPI

You can also use the fine-tuned model for inference via a FastAPI application. Run the `app.py` script:

This will start the FastAPI server at `http://localhost:8080`. You can send a POST request to `http://localhost:8080/symptoms` with either form data or JSON data to generate text based on the provided input string.

##### Form Data

To send form data, make a POST request to `http://localhost:8080/generate` with the form data containing the `input_str` field.

##### JSON Data

To send JSON data, make a POST request to `http://localhost:8080/generate` with a JSON payload like this:

```json
{
    "disease_name": "Kidney Failure"
}

Both requests will return the generated text as a JSON response with the Symptoms key.
```

## Dependencies

* Python 3.6+
* PyTorch
* Hugging Face Transformers
* Datasets
* pandas
* tqdm
* FastAPI

## Configuration

The project uses a config.py file to store the paths to the locally saved tokenizer and pre-trained model files. Create a config.py file in the project directory with the following contents:

```python
TOKENIZER_PATH = 'path/to/tokenizer'
MODEL_PATH = 'path/to/model.pt'
```

Replace 'path/to/tokenizer' and 'path/to/model.pt' with the actual paths to your locally saved tokenizer and pre-trained model files, respectively.

## Files

* data.py : Thi sscript is used to preprocess the dataset before training the model.
* train.py: Contains the code for fine-tuning the model.
* inference.py: Contains the code for generating text using the fine-tuned model.
* app.py: Contains the FastAPI application for inference.
* config.py: Stores the paths to the locally saved tokenizer and pre-trained model files.
* README.md: This file, providing an overview of the project.

## Acknowledgments

The "Diseases_Symptoms" dataset is from the Hugging Face datasets repository.
The DistilGPT-2 model is provided by the Hugging Face Transformers library.