import os
from dotenv import load_dotenv, find_dotenv

# load env BEFORE importing modules that construct the Groq client
load_dotenv(find_dotenv())

import streamlit as st
from nodes import graph
from core.doc_processor import DocProcessor
from core.vectorstore import VectorStoreQdrant
import nomic


key = os.getenv("NOMIC_API")
nomic.cli.login(key)

agent = graph.create_graph()

agent.get_graph().print_ascii()

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# prompt = st.chat_input("What's up")
# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.status("Running pipeline"):
#         result = agent.invoke({
#             "query": prompt,
#             "intent": "",
#             "weather_data": {},
#             "pdf_context": "",
#             "final_response": ""
#         })

#     with st.chat_message("assistant"):
#         st.markdown(result["final_response"])
#         st.session_state.messages.append({"role": "assistant", "content": result["final_response"]})


import streamlit as st

# --- session init ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- render history ----------
for message in st.session_state.messages:
    # message["role"] should be "user" or "assistant"
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- input ----------
prompt = st.chat_input("What's up")

if prompt and not st.session_state.processing:
    # mark we're processing (prevents race on reruns)
    st.session_state.processing = True

    # persist user message and show it immediately
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # run pipeline and capture result
    try:
        with st.status("Running pipeline"):
            result = graph.invoke_runnable(agent, prompt, "", {}, "", "")
            # result = agent.invoke({
            #     "query": prompt,
            #     "intent": "",
            #     "weather_data": {},
            #     "pdf_context": "",
            #     "final_response": ""
            # })
    except Exception as e:
        assistant_text = f"Pipeline error: {e}"
    else:
        # robustly extract assistant text from common keys
        if isinstance(result, dict):
            assistant_text = result.get("final_response") or result.get("answer") or result.get("text") or ""
        else:
            # fallback if result is a string or other type
            assistant_text = str(result)

        # final fallback to ensure non-empty output
        if not assistant_text:
            assistant_text = "(no assistant text returned)"

    # display assistant and persist
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    # done processing
    st.session_state.processing = False
