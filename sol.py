import json
import os
from typing import List
from PyPDF2 import PdfReader, PdfWriter
import solara
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Set up Hugging Face API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# Merge PDFs
def merge_pdfs(pdf_directory: str, output_pdf: str) -> None:
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"The directory '{pdf_directory}' does not exist.")
    if os.path.exists(output_pdf):
        print(f"The merged PDF already exists at {output_pdf}. No need to create it again.")
        return
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory '{pdf_directory}'.")
    pdf_writer = PdfWriter()
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    with open(output_pdf, 'wb') as output:
        pdf_writer.write(output)
    print(f"Merged PDF saved as {output_pdf}")

# Load configuration from JSON file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

pdf_directory = config.get("pdf_directory")

# Validate the directory
if not os.path.exists(pdf_directory):
    raise FileNotFoundError(f"The directory '{pdf_directory}' does not exist.")

# Directory and merged file path
merged_pdf_path = os.path.join(pdf_directory, 'merged_document.pdf')
merge_pdfs(pdf_directory, merged_pdf_path)

# Load Merged PDF and Split into Chunks
loader = PyPDFLoader(merged_pdf_path)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
splits = text_splitter.split_documents(pages)
print(f"Number of document chunks created: {len(splits)}")

# Initialize Vector Store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = './chroma_db'

try:
    if os.path.exists(persist_directory):
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        print(f"Using existing vector database with {vectordb._collection.count()} vectors.")
    else:
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
        print(f"Created new vector database with {vectordb._collection.count()} vectors.")
except Exception as e:
    print(f"Error initializing vector database: {e}")
    raise

# Use HuggingFaceHub for inference
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5, "max_new_tokens": 250})

# Improved Prompt Template
template = """
You are a knowledgeable and concise assistant. Use the provided context to answer the user's question directly and accurately. Avoid redundancy and unnecessary information.

### Context:
{context}

### Question:
{question}

### Response:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Create RetrievalQA Chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=False,  # Do not return source documents
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
except Exception as e:
    print(f"Error creating QA chain: {e}")
    raise

# Solara App
@solara.component
def App():
    query, set_query = solara.use_state("")  # User's current query
    chat_history, set_chat_history = solara.use_state([])  # Full chat history
    
    def handle_submit():
        if query.strip():  # Only process non-empty queries
            try:
                # Retrieve response from the backend
                response = qa_chain.invoke({"query": query.strip()})
                # Extract the meaningful part of the response
                raw_response = response.get('result', '').strip()
                clean_response = raw_response.split("Response:")[-1].strip() if "Response:" in raw_response else raw_response

                 # Update chat history with user query and bot response
                set_chat_history(chat_history + [("You", query.strip()), ("Bot", clean_response)])
                set_query("")  # Clear input for the next query
            except Exception as e:
                # Handle errors gracefully
                set_chat_history(chat_history + [("You", query.strip()), ("Bot", f"An error occurred: {e}")])
                set_query("")  # Allow further queries after an error
    
         
    # Full-Screen Layout
    with solara.Column(
        style={
            "height": "100vh",
            "width": "100%",
            "backgroundColor": "#F5F6FA",
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "0",
            "margin": "0",
        }
    ):
        # Header
        with solara.Card(
            style={
                "width": "100%",
                "padding": "25px",
                "backgroundColor": "#4A90E2",
                "color": "#FFFFFF",
                "textAlign": "center",
                "fontSize": "34px",
                "fontWeight": "bold",
                "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.1)",
            }
        ):
            solara.Markdown("💬 **ALLIANZ CAR INSURANCE CHAT BOT** ")

        # Chat History Section
        with solara.Column(
            style={
                "flexGrow": "1",
                "width": "100%",
                "padding": "20px",
                "overflowY": "auto",  # Enable scrolling for chat history
                "backgroundColor": "#FFFFFF",
                "scrollBehavior": "smooth",
            }
        ):
            for speaker, text in chat_history:
                if speaker == "You":
                    # User Message
                    with solara.Row(
                        style={
                            "justifyContent": "flex-end",
                            "marginBottom": "10px",
                        }
                    ):
                        with solara.Card(
                            style={
                                "backgroundColor": "#D0E8FF",
                                "borderRadius": "15px",
                                "padding": "10px 15px",
                                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                                "maxWidth": "60%",
                            }
                        ):
                            solara.Markdown(f"**{speaker}:** {text}", style={"color": "#004085", "fontSize": "15px"})
                else:
                    # Bot Message
                    with solara.Row(
                        style={
                            "justifyContent": "flex-start",
                            "marginBottom": "10px",
                        }
                    ):
                        with solara.Card(
                            style={
                                "backgroundColor": "#E3FCEC",
                                "borderRadius": "15px",
                                "padding": "10px 15px",
                                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                                "maxWidth": "60%",
                            }
                        ):
                            solara.Markdown(f"**{speaker}:** {text}", style={"color": "#225522", "fontSize": "15px"})

        # Input Section
        with solara.Row(
            style={
                "width": "100%",
                "padding": "15px",
                "alignItems": "center",
                "backgroundColor": "#F7F8FA",
                "borderTop": "1px solid #E5E7EB",
                "boxShadow": "0px -2px 4px rgba(0, 0, 0, 0.1)",
            }
        ):
            solara.InputText(
                value=query,
                on_value=set_query,
                label="Type your message...",
                style={
                    "flexGrow": "1",
                    "padding": "10px",
                    "border": "1px solid #CCCCCC",
                    "borderRadius": "20px",
                    "marginRight": "10px",
                },
            )
            solara.Button(
                "Send",
                on_click=handle_submit,
                style={
                    "padding": "10px 20px",
                    "backgroundColor": "#4CAF50",
                    "color": "#FFFFFF",
                    "border": "none",
                    "borderRadius": "20px",
                    "cursor": "pointer",
                    "fontWeight": "bold",
                    "fontSize": "16px",
                },
            )

# Alias App as Page for Solara CLI
Page = App
