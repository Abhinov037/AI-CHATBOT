import os
import json
from PIL import Image
import google.generativeai as genai

# working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# path of config_data file
config_file_path = f"{working_dir}/config.json"
config_data = json.load(open(config_file_path))

# loading the GOOGLE_API_KEY
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)

# Load Gemini-1.5-Flash Model
def load_gemini_1_5_flash_model():
    gemini_1_5_flash_model = genai.GenerativeModel("gemini-1.5-flash")
    return gemini_1_5_flash_model

# Response from Gemini-1.5-Flash Model (Text to Text)
def gemini_1_5_flash_response(user_prompt):
    gemini_1_5_flash_model = load_gemini_1_5_flash_model()
    response = gemini_1_5_flash_model.generate_content(user_prompt)
    return response.text

# Response from Gemini-1.5-Flash Vision Model (Image to Text)
def gemini_1_5_flash_vision_response(prompt, image):
    gemini_1_5_flash_vision_model = genai.GenerativeModel("gemini-1.5-flash-vision")
    response = gemini_1_5_flash_vision_model.generate_content([prompt, image])
    return response.text

# Embedding model for text-to-embedding
def embeddings_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list

