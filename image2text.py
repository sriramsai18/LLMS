import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Set page title
st.set_page_config(page_title="Image 2 Text Generator", layout="centered")

# App title
st.title("🖼️ Image 2 Text with BLIP")

# Load the BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast = False)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    #prompt = st.text_input("Prompt Template ", value="Imagine you are pro at generating captions for images any kind of images. NOW, Describe this image in a natural and detailed way.")
    # Generate caption button
    if st.button("Generate Text"):
        with st.spinner("Generating Text.."):
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
        st.success("Generated Text According to Image:")
        st.write(f"📢 **{caption}**")
