import os
import logging
import time

# Use python-dotenv to load the API key from a .env file
from dotenv import load_dotenv

# LangChain and Gradio components
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import gradio as gr

# --- Basic Setup ---
# Load environment variables from a .env file
load_dotenv() 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Create a folder named "my_documents" and place your PDFs inside it.
PDF_DIRECTORY = "my_documents" 
# This is where the processed, searchable database will be stored.
VECTOR_STORE_PATH = "my_vector_db"

class RAGSystem:
    """
    Handles the local RAG system. If a vector store exists, it loads it.
    If not, it builds it from your PDFs in the "my_documents" directory using batch processing.
    """
    def __init__(self, store_path, pdf_directory, embedding_model):
        self.store_path = store_path
        self.pdf_directory = pdf_directory
        self.embedding_model = embedding_model

        if not os.path.exists(self.pdf_directory):
            logging.info(f"Creating directory for your documents at: {self.pdf_directory}")
            os.makedirs(self.pdf_directory)

        if not os.path.exists(self.store_path):
            logging.info(f"Vector store not found. Building it from documents in '{self.pdf_directory}'...")
            self._build_from_source()
        else:
            logging.info(f"Loading existing vector store from: {self.store_path}")

        try:
            self.vector_store = Chroma(
                persist_directory=self.store_path,
                embedding_function=self.embedding_model
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            logging.info("RAG system initialized successfully.")
        except Exception as e:
            logging.error(f"FATAL: Failed to load Chroma database. If you added new files, try deleting the '{self.store_path}' folder and restarting. Error: {e}")
            raise e

    def _build_from_source(self):
        """Processes all PDF files and builds the Chroma database in batches."""
        logging.info("Starting to build vector store from your documents...")
        all_docs = []
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logging.warning(f"No PDF files found in '{self.pdf_directory}'. The AI will not have any custom knowledge.")
            os.makedirs(self.store_path, exist_ok=True)
            return

        for file_name in pdf_files:
            file_path = os.path.join(self.pdf_directory, file_name)
            try:
                loader = PyMuPDFLoader(file_path)
                all_docs.extend(loader.load())
                logging.info(f"Loaded {file_name}")
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
        
        if not all_docs:
            logging.warning("No documents could be loaded. The AI will not have any custom knowledge.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        logging.info(f"Split your documents into {len(splits)} text chunks. Now creating the searchable database...")

        # Process in batches to handle large numbers of documents gracefully
        batch_size = 100 
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            try:
                if i == 0:
                    # For the first batch, create the database
                    Chroma.from_documents(
                        documents=batch,
                        embedding=self.embedding_model,
                        persist_directory=self.store_path
                    )
                else:
                    # For subsequent batches, load the existing instance and add to it
                    instance = Chroma(persist_directory=self.store_path, embedding_function=self.embedding_model)
                    instance.add_documents(documents=batch)
                
                progress = min(i + batch_size, len(splits))
                logging.info(f"Embedded and added documents {progress}/{len(splits)}")
            except Exception as e:
                logging.error(f"Error embedding batch starting at index {i}: {e}. Skipping batch.")
                time.sleep(2) # Wait a moment before trying the next batch
        
        logging.info("Finished building the vector store. It is now saved locally.")

    def query(self, question: str):
        """Searches your documents for relevant context."""
        if not hasattr(self, 'retriever') or not self.retriever:
            return "RAG system is not initialized. Make sure you have PDFs in the 'my_documents' folder."
        logging.info(f"Searching documents for: '{question}'")
        retrieved_docs = self.retriever.invoke(question)
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        return context

class PersonalAgent:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file! Please see the README for instructions.")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.google_api_key)
        self.rag_system = RAGSystem(VECTOR_STORE_PATH, PDF_DIRECTORY, embeddings)

    def get_system_prompt(self, rag_context: str):
        """Generates the main instruction prompt for the AI."""
        return (
            "You are a helpful AI assistant. Your persona should be professional and engaging."
            "\n\n## Context from Personal Documents:\n"
            "The following information is sourced from the user's personal documents. "
            "Use this context to provide an accurate and effective response. "
            "Synthesize the information seamlessly into your answer without mentioning the act of retrieving documents."
            f"\n---\n{rag_context}\n---\n\n"
            "If you don't know an answer based on the provided context, simply state that the topic is outside your current knowledge base."
        )

    def chat(self, message: str, history: list):
        """Handles the main chat logic."""
        logging.info(f"Received user query: '{message}'")
        rag_context = self.rag_system.query(message)
        system_prompt = self.get_system_prompt(rag_context)
        
        # Combine system instructions, chat history, and the new message
        # Gradio's history format is a list of tuples [(user, assistant), ...], we need to convert it
        formatted_history = []
        for user_msg, assistant_msg in history:
            formatted_history.append({"role": "user", "content": user_msg})
            formatted_history.append({"role": "assistant", "content": assistant_msg})

        full_prompt = [{"role": "system", "content": system_prompt}] + formatted_history + [{"role": "user", "content": message}]

        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=self.google_api_key)
            response = llm.invoke(full_prompt)
            return response.content
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return "Sorry, an unexpected error occurred. Please check the console for details."

# --- Gradio UI ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: The GOOGLE_API_KEY is not set in your .env file.")
        print("Please follow the instructions in the README.md file to set it up.")
    else:
        agent = PersonalAgent()
        
        with gr.Blocks(theme=gr.themes.Monochrome(), css="#chatbot { min-height: 400px; }") as demo:
            gr.Markdown(
                """
                # Chat with Your Personal AI Assistant
                Ask me anything about the information contained in your documents.
                """
            )
            chatbot = gr.Chatbot(elem_id="chatbot", label="AI Assistant", bubble_full_width=False, height=500)
            with gr.Row():
                msg = gr.Textbox(label="Your Message", placeholder="Type your question here...", scale=4, container=False)
            clear_button = gr.ClearButton([msg, chatbot], value="Clear Conversation")
            
            def respond(message, chat_history):
                bot_message = agent.chat(message, chat_history)
                chat_history.append((message, bot_message))
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
        demo.launch()