import streamlit as st
from bot import initialize_rag_pipeline
from langchain.schema import SystemMessage, HumanMessage


# Configure page
st.set_page_config(
    page_title="Persian ESL Tutor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat-like interface
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
    border: 1px solid #4e5561;
}
.chat-message.assistant {
    background-color: #383f4d;
    border: 1px solid #4e5561;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 100%;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("🧠 معلم زبان انگلیسی ایرانی")
st.markdown("سوال خود را درباره یادگیری انگلیسی بپرسید:")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize models once
if "db" not in st.session_state or "llm" not in st.session_state:
    with st.spinner("Loading models and index..."):
        db, llm = initialize_rag_pipeline()
        st.session_state.db = db
        st.session_state.llm = llm

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
query = st.chat_input("سوال خود را وارد کمید: ")

# Process user input
if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("درحال پاسخ دادن..."):
            # Get relevant documents
            docs = st.session_state.db.similarity_search(query, k=4)
            retrieved_content = "\n\n".join(doc.page_content for doc in docs)

            # Compose prompt
            prompt = f"""You are a Persian tutor whose students are Persian.
Use the following excerpts from an ESL textbook to answer the question.
Keep a normal conversation until someone asks you for an English tutorial. Provide a detailed explanation.
Don't explain anything unless they want you to. Answer the question in Persian.
Don't say anything in English unless it's about teaching something.

Context:
{retrieved_content}

Question:
{query}
"""

            # Get previous messages for context if available
            previous_messages = []
            for msg in st.session_state.messages[-5:]:  # Last 5 messages for context
                if msg["role"] == "user":
                    previous_messages.append(HumanMessage(content=msg["content"]))
                else:
                    previous_messages.append(SystemMessage(content=msg["content"]))

            # Use LangChain's structured message objects
            response = st.session_state.llm([
                SystemMessage(content="You are a helpful English tutor for ESL learners."),
                *previous_messages,
                HumanMessage(content=prompt)
            ])

            # Display response
            st.markdown(response.content)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.content})

# Add a clear chat button in the sidebar
st.sidebar.title("Chat Controls")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Add a more prominent clear chat button below the chat input
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
