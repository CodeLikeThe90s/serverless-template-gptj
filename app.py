from transformers import pipeline
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    print("Active Device:{}".format(device))
    model = pipeline('text-generation', model='EleutherAI/gpt-j-6B', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    max_length = model_inputs.get('max_length', None)
    min_length = model_inputs.get('min_length', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    if max_length == None:
        return {'message': "No max_length provided"}
    if min_length == None:
        return {'message': "No min_length provided"}
    
    # Run the model
    result = model(prompt, max_length=max_length, min_length=min_length, do_sample=False)[0]

    # Return the results as a dictionary
    return result
