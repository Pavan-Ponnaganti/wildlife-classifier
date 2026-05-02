import streamlit as st
import torch
from torchvision import transforms
import timm
from PIL import Image
import json
import os

# Set page config
st.set_page_config(page_title="Indian Wildlife & Bird Identifier", page_icon="🐾")

st.title("🐾 Indian Wildlife & Bird Identifier")
st.write("Upload a photo of an animal, and we'll tell you what it is!")

# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_model():
    if not os.path.exists('classes.json'):
        return None, None
    with open('classes.json', 'r') as f:
        class_names = json.load(f)
        
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(class_names))
    
    if os.path.exists('efficientnet_b0_animals.pth'):
        model.load_state_dict(torch.load('efficientnet_b0_animals.pth', map_location=torch.device('cpu')))
    else:
        st.warning("Could not find the trained model weights 'efficientnet_b0_animals.pth'. Using untrained weights for demonstration. Please run train.py first to fine-tune the model.")
        
    model.eval()
    return model, class_names

model, class_names = load_model()

@st.cache_data
def load_animal_info():
    if os.path.exists('animal_info.json'):
        with open('animal_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

animal_info = load_animal_info()

# Preprocessing transforms (same as val_transforms in train.py)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("Model or classes.json not found. Please ensure train.py has been executed to map classes.")
    else:
        # Display the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        st.write("Analyzing...")
        # Preprocess and predict
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model
        
        with torch.no_grad():
            output = model(input_batch)
        
        # Get top 3 probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        
        st.subheader("Predictions:")
        for i in range(top3_prob.size(0)):
            class_idx = top3_catid[i].item()
            animal_name = class_names[class_idx]
            prob = top3_prob[i].item() * 100
            
            # UI presentation for each prediction
            with st.expander(f"{i+1}. {animal_name.capitalize()} ({prob:.2f}%)", expanded=(i==0)):
                info = animal_info.get(animal_name, None)
                if info:
                    st.write(f"**Wikipedia Summary:** {info.get('description', 'N/A')}")
                    st.write(f"**IUCN Status:** `{info.get('iucn_status', 'N/A')}`")
                else:
                    st.write("No additional facts found for this animal.")
                    
st.markdown("---")
st.caption("Built for SMAI Assignment 3 - T7.6 Indian Wildlife Identifier")
