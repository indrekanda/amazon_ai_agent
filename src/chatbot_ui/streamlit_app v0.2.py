import streamlit as st
from openai import OpenAI
from groq import Groq
from google import genai
from qdrant_client import QdrantClient

from core.config import config
from retrieval import rag_pipeline

# works with retrieval v0.1
# It is not localhost, because we are running the app from docker compose, 
# If we are running locally, we can use localhost:6333
qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")


## A sidebar with a dropdown for the model list and providers
with st.sidebar:
    st.title("Settings")

    #Dropdown for model
    provider = st.selectbox("Provider", ["Groq", "Google", "OpenAI"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    elif provider == "Groq":
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])
    else:
        model_name = st.selectbox("Model", ["gemini-2.0-flash"])
    
    # Add other settings
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=1000, value=500, step=100)

    # Save provider, model, temperature, and max_tokens to session state
    st.session_state.provider = provider
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens


# Initialize the client
if st.session_state.provider == "OpenAI":
    client = OpenAI(api_key=config.OPENAI_API_KEY)
elif st.session_state.provider == "Groq":
    client = Groq(api_key=config.GROQ_API_KEY)
else:
    client = genai.Client(api_key=config.GOOGLE_API_KEY)


# Response generation function
def run_llm(client, messages, max_tokens=500, temperature=1):
    if st.session_state.provider == "Google":
        return client.models.generate_content(
            model=st.session_state.model_name,
            contents=[message["content"] for message in messages],
            config=genai.types.GenerateContentConfig( # types is correct
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
        ).text
    else:
        return client.chat.completions.create( 
            model=st.session_state.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        ).choices[0].message.content


# Initialize the messages
if "messages" not in st.session_state:
    st.session_state.messages = [
       # {"role": "system", "content": "You should never disclose what model are you based on"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # output = run_llm(client, st.session_state.messages)
        output = rag_pipeline(prompt, qdrant_client)
        st.write(output["answer"])
    st.session_state.messages.append({"role": "assistant", "content": output})