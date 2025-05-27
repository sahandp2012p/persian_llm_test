mport os
import fitz  # PyMuPDF for PDF processing
import warnings
import pygame  # For playing audio files
import time  # For sleep function
import speech_recognition as sr  # For speech-to-text

from dotenv import load_dotenv
load_dotenv()

# Suppress warnings to keep the console clean
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# LangChain and HuggingFace imports for RAG pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
import openai

os.environ["SDL_AUDIODRIVER"] = "alsa"  # or "alsa"

# --- Configuration ---
PDF_PATH = "Complete English All-in-One for ESL Learners Book.pdf"
TEXT_PATH = "esl_book.txt"
FAISS_FOLDER = "esl_book_faiss_index"



def speak(text):
    try:
        print("🔊 Generating audio...")

        response = openai.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=text
        )

        with open("output.mp3", "wb") as f:
            f.write(response.content)

        print("🔊 Playing audio...")
        pygame.mixer.init()
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.quit()
        os.remove("output.mp3")
        print("✅ Audio playback complete.")
    except Exception as e:
        print(f"❌ Error using Voice: {e}")


def listen():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("🎤 در حال گوش دادن... لطفاً صحبت کنید.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)  # listens until silence detected

    try:
        print("📝 در حال تبدیل صدا به متن...")
        text = recognizer.recognize_google(audio, language="fa-IR")
        print(f"✅ متن تشخیص داده شده: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ متوجه نشدم. لطفاً دوباره تلاش کنید.")
        return ""
    except sr.RequestError as e:
        print(f"❌ خطا در ارتباط با سرویس تشخیص صدا: {e}")
        return ""


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

    print("\n--- ESL Tutor Ready ---")
    print("با صدای خود یا تایپ کردن سوال بپرسید. برای خروج بنویسید یا بگویید 'exit' یا 'خروج'.")

    while True:
        use_voice = input("\n🗣 آیا می‌خواهید با صدا سوال بپرسید؟ (y/n): ").strip().lower()
        if use_voice == "y":
            query = listen()
        else:
            query = input("\n❓ سوال خود را بپرسید (یا 'exit' را تایپ کنید): ")

        if query.lower() in ["exit", "quit", "خروج"]:
            print("👋 خداحافظ!")
            break

        if not query.strip():
            print("⚠️ لطفاً یک سوال معتبر وارد کنید.")
            continue

        try:
            print("🔍 Searching for relevant information...")
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

            print("🧠 Getting answer from LLM...")
            response = llm([
                SystemMessage(content="You are a helpful English tutor for ESL learners."),
                HumanMessage(content=prompt)
            ])

            llm_response_text = response.content
            print("\n✅ پاسخ:\n", llm_response_text)
            speak(llm_response_text)

        except Exception as e:
            print(f"❌ خطایی رخ داد: {e}")
            print("لطفاً دوباره تلاش کنید یا سوال دیگری بپرسید.")
