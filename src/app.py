import streamlit as st
from graph import ask_rag

st.set_page_config(
    page_title="Bot",
    page_icon="🤖",
    layout="wide",
)

st.title("Agentic RAGbot")
st.write("Let's get started")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            answer = ask_rag(user_question)

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
