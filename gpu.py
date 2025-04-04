import os
import streamlit as st
import pathlib

lib_path = pathlib.Path("lib/libllama.so")
st.write("Does libllama.so exist at runtime?", lib_path.exists())
st.write("Absolute path:", lib_path.resolve())

# Force override to absolute path of your lib
import llama_cpp.llama_cpp as llama_core
llama_core._override_base_path = "/mount/src/streamlit-llama-cpp/lib"

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
# Load the LLM from GGUF file
repo_id = "Rudrresh/cdoeforces-llama-gguf"
odel_file = "llama-3-3b-coder.gguf"
model_path = hf_hub_download(repo_id = repo_id, filename=model_file)
# n_threads
llm = Llama(model_path=model_path,n_gpu_layers=30,n_ctx=512,temperature=0.2,repeat_penalty=1.1,top_k_sampling=40,top_p_sampling=0.95,min_p_sampling=0.05)
def generate_llm_response(prompt):
    output = llm(prompt, max_tokens=1024)
    return output["choices"][0]["text"]

import streamlit as st
#import speech_recognition as sr
import numpy as np

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# User input (text)
st.title("LeetCode Practice LLM")
user_input = st.chat_input("Type a message or use voice...")


# Process response
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Get response from GGUF LLM
    response = generate_llm_response(user_input)

    # Display response
    st.chat_message("assistant").write(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})

