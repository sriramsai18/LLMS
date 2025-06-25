import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "TinyLlama/TinyLlama-1.1B-chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

st.markdown("<h1 style='text-align: center;'>TURBO CHAT⚡</h1>", unsafe_allow_html=True)
st.write("<h6 style='text-align:center;'>Turbo Chat is a Q&A chatbot.</h6>",unsafe_allow_html=True)

user_input = st.text_input("Ask Anything...!")

if user_input:
    with st.spinner("Generating..."):
        tokenizer, model = load_model_and_tokenizer()
        chat_prompt = f"<|user|>\n{user_input}\n<|Assistant|>\n"
        
        inputs = tokenizer(chat_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            full_output = tokenizer.decode(output[0], skip_special_tokens=True)
            response = full_output[len(chat_prompt):].strip()

        st.subheader("Response")
        st.write(response)