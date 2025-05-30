import streamlit as st
from bot import initialize_rag_pipeline
from langchain.schema import SystemMessage, HumanMessage  # ✅ Import proper message classes

st.set_page_config(page_title="Persian ESL Tutor", layout="wide")

st.title("🧠 ESL Tutor for Persian Speakers")
st.markdown("Type your question about English learning below:")

# ✅ Initialize models once
if "db" not in st.session_state or "llm" not in st.session_state:
    with st.spinner("Loading models and index..."):
        db, llm = initialize_rag_pipeline()
        st.session_state.db = db
        st.session_state.llm = llm

# ✅ User input
query = st.text_input("سوال خود را وارد کمید: ", key="input")

# ✅ Run when button is pressed
if st.button("ارسال سوال") and query:
    with st.spinner("درحال پاسخ دادن..."):
        docs = st.session_state.db.similarity_search(query, k=4)
        retrieved_content = "\n\n".join(doc.page_content for doc in docs)

        # ✅ Compose prompt
        prompt = f"""You are a Persian tutor whose students are Persian.
Use the following excerpts from an ESL textbook to answer the question.
Keep a normal conversation until someone asks you for an English tutorial. Provide a detailed explanation.
Don't explain anything unless they want you to. Answer the question in Persian.
Don't say anything in English unless it’s about teaching something.

Context:
{retrieved_content}

Question:
{query}
"""

        # ✅ Use LangChain's structured message objects
        response = st.session_state.llm([
            SystemMessage(content="You are a helpful English tutor for ESL learners."),
            HumanMessage(content=prompt)
        ])

        # ✅ Show response
        st.success("✅ پاسخ آماده شد:")
        st.markdown(response.content)
