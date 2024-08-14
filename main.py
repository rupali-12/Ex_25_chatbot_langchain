import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# Set up the model
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

# Create a store for message histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the model with message history
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Streamlit UI
def main():
    st.title("Chatbot Interface")

    # Initialize session state if not already
    if "session_id" not in st.session_state:
        st.session_state.session_id = "chat1"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_continue_options" not in st.session_state:
        st.session_state.show_continue_options = False

    # Session ID input
    session_id = st.text_input("Enter Session ID:", st.session_state.session_id)
    st.session_state.session_id = session_id

    # Display chat history
    st.markdown("### Chat History")
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.markdown(
                f'<div style="background-color:#d4edda;border:1px solid #c3e6cb;border-radius:8px;padding:10px;margin:5px;">**You:** {message.content}</div>', 
                unsafe_allow_html=True
            )
        elif isinstance(message, AIMessage):
            st.markdown(
                f'<div style="background-color:#e2e3e5;border:1px solid #d6d6d6;border-radius:8px;padding:10px;margin:5px;">**Chatbot:** {message.content}</div>', 
                unsafe_allow_html=True
            )

    # Input field for new messages
    if not st.session_state.show_continue_options:
        user_input = st.text_input("You:", key="user_input", placeholder="Type your message here...")

        if st.button("Send"):
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append(HumanMessage(content=user_input))

                # Get the chatbot's response
                response = with_message_history.invoke(
                    [HumanMessage(content=user_input)],
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )

                # Add chatbot response to chat history
                st.session_state.messages.append(AIMessage(content=response.content))

                # Show continuation options
                st.session_state.show_continue_options = True
            else:
                st.warning("Please enter a message.")
    else:
        st.markdown("### Do you want to continue chatting?")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Continue Chat"):
                # Reset state for new chat
                st.session_state.show_continue_options = False
                st.session_state.user_input = ""  # Clear the input field
        with col2:
            if st.button("Clear History"):
                # Clear chat history and reset state
                st.session_state.messages = []
                st.session_state.show_continue_options = False
                st.session_state.user_input = ""  # Clear the input field
                st.success("Session history cleared.")
    
    # Show goodbye message if not continuing
    if st.session_state.show_continue_options and len(st.session_state.messages) == 0:
        st.markdown("### Goodbye! Have a great day! 😊")

if __name__ == "__main__":
    main()







