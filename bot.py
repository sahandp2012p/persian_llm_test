import os
import fitz  # PyMuPDF for PDF processing
import warnings

from dotenv import load_dotenv
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# LangChain and HuggingFace imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
import openai

# --- Configuration ---
PDF_PATH = "Complete English All-in-One for ESL Learners Book.pdf"
TEXT_PATH = "esl_book.txt"
FAISS_FOLDER = "esl_book_faiss_index"


def initialize_rag_pipeline():
    print("--- Initializing RAG components ---")

    api_key = os.getenv("KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please create a .env file with KEY=your-openai-api-key.")
    os.environ["OPENAI_API_KEY"] = api_key

    text = ""
    if os.path.exists(TEXT_PATH):
        print("✅ Loaded text from existing file.")
        with open(TEXT_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    elif os.path.exists(PDF_PATH):
        print("📄 Extracting text from PDF...")
        try:
            doc = fitz.open(PDF_PATH)
            text = "".join([page.get_text() for page in doc])
            doc.close()
            with open(TEXT_PATH, "w", encoding="utf-8") as f:
                f.write(text)
            print("✅ Text extracted and saved.")
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            raise
    else:
        raise FileNotFoundError(f"PDF file '{PDF_PATH}' not found.")

    print("🔹 Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Generated {len(chunks)} text chunks.")

    print("🔄 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Embedding model loaded.")
    index_file = os.path.join(FAISS_FOLDER, "index.faiss")
    pkl_file = os.path.join(FAISS_FOLDER, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        print("📦 Loading existing FAISS index...")
        try:
            db = FAISS.load_local(FAISS_FOLDER, embedding_model, allow_dangerous_deserialization=True)
            print(f"✅ FAISS index loaded from {FAISS_FOLDER}/")
        except Exception as e:
            print(f"Error loading FAISS index, recreating: {e}")
            db = FAISS.from_texts(chunks, embedding_model)
            db.save_local(FAISS_FOLDER)
            print(f"✅ New FAISS index created and saved to {FAISS_FOLDER}/")
    else:
        print("📦 Creating and saving new FAISS index...")
        db = FAISS.from_texts(chunks, embedding_model)
        db.save_local(FAISS_FOLDER)
        print(f"✅ FAISS index saved to {FAISS_FOLDER}/")

    print("🤖 Initializing chat model...")
    llm = init_chat_model("gpt-3.5-turbo-0125", model_provider="openai")
    print("✅ Chat model initialized.")

    print("--- RAG components initialized successfully ---")
    return db, llm


# --- Main execution block ---
if __name__ == '__main__':
    db, llm = None, None
    try:
        db, llm = initialize_rag_pipeline()
    except Exception as e:
        print(f"Fatal error during initialization: {e}")
        print("Please resolve the issues and restart the script.")
        exit()

    # Initialize conversation memory
    memory = ConversationBufferMemory(return_messages=True)

    print("\n--- ESL Tutor Ready ---")
    print("❓ سوال خود را تایپ کنید. برای خروج بنویسید 'exit' یا 'خروج'.")

    while True:
        query = input("\n❓ سوال خود را وارد کنید: ").strip()

        if query.lower() in ["exit", "quit", "خروج"]:
            print("👋 خداحافظ!")
            break

        if not query:
            print("⚠️ لطفاً یک سوال معتبر وارد کنید.")
            continue

        try:
            print("🔍 جستجوی اطلاعات مرتبط...")
            docs = db.similarity_search(query, k=4)
            retrieved_content = "\n\n".join(doc.page_content for doc in docs)

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

            # Load conversation history from memory
            history = memory.load_memory_variables({}).get("history", [])

            # Combine past messages + new human prompt
            messages = history + [HumanMessage(content=prompt)]

            print("🧠 دریافت پاسخ از مدل زبانی...")
            response = llm(messages)

            # Save new interaction to memory
            memory.save_context(
                {"input": prompt},
                {"output": response.content}
            )

            print("\n✅ پاسخ:\n", response.content)

        except Exception as e:
            print(f"❌ خطایی رخ داد: {e}")
            print("لطفاً دوباره تلاش کنید یا سوال دیگری بپرسید.")
