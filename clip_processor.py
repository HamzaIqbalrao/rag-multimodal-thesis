import torch
from torchvision import transforms
from PIL import Image
import os
import json
from transformers import CLIPModel, CLIPProcessor


model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()

# Set up image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
#   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to create embedding from an image
def create_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            image_features = outputs.pooler_output
        else:
            image_features = outputs

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().flatten()   

def create_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

        # If transformers returns an object instead of tensor
        if hasattr(outputs, "pooler_output"):
            text_features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            text_features = outputs.last_hidden_state[:, 0, :]
        else:
            text_features = outputs  # already tensor

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().flatten()


def add_embeddings(json_file):
    with open('images_metadata/' + json_file, 'r') as file:
        data = json.load(file)

        text = data['generated_description']
        image_file = data['image_filename']

        file.close()

    text_embedding = create_text_embedding(text)
    image_embedding = create_image_embedding('images_metadata/'+image_file)

    data['text_embedding'] = text_embedding.tolist()
    data['image_embedding'] = image_embedding.tolist()

    # prune fields on the metadata

    allowed_keys = ["photo_id", "title", "description", "geolocation", "image_filename",
                    "generated_description", "text_embedding", "image_embedding"]
    pruned_data = {key: data[key] for key in allowed_keys}

    return pruned_data

