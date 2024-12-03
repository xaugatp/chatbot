# import os
# from typing import List, Any, Optional
# from PyPDF2 import PdfReader, PdfWriter
# import solara
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceHub

# # Set up Hugging Face API token
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# # Merge PDFs
# def merge_pdfs(pdf_directory: str, output_pdf: str) -> None:
#     if not os.path.exists(pdf_directory):
#         raise FileNotFoundError(f"The directory '{pdf_directory}' does not exist.")
#     if os.path.exists(output_pdf):
#         print(f"The merged PDF already exists at {output_pdf}. No need to create it again.")
#         return
#     pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
#     if not pdf_files:
#         raise FileNotFoundError(f"No PDF files found in directory '{pdf_directory}'.")
#     pdf_writer = PdfWriter()
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(pdf_directory, pdf_file)
#         try:
#             pdf_reader = PdfReader(pdf_path)
#             for page in pdf_reader.pages:
#                 pdf_writer.add_page(page)
#         except Exception as e:
#             print(f"Error processing {pdf_file}: {e}")
#     with open(output_pdf, 'wb') as output:
#         pdf_writer.write(output)
#     print(f"Merged PDF saved as {output_pdf}")

# # Directory and merged file path
# pdf_directory = "C:/Users/sauga/Desktop/NLP303/Assessment 3/"
# merged_pdf_path = os.path.join(pdf_directory, 'merged_document.pdf')
# merge_pdfs(pdf_directory, merged_pdf_path)

# # Load Merged PDF and Split into Chunks
# loader = PyPDFLoader(merged_pdf_path)
# pages = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
# splits = text_splitter.split_documents(pages)
# print(f"Number of document chunks created: {len(splits)}")

# # Initialize Vector Store
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# persist_directory = './chroma_db'

# try:
#     if os.path.exists(persist_directory):
#         vectordb = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embedding,  # Pass the embedding object directly
#         )
#         print(f"Using existing vector database with {vectordb._collection.count()} vectors.")
#     else:
#         vectordb = Chroma.from_documents(
#             documents=splits,
#             embedding=embedding,  # Pass the embedding object directly
#             persist_directory=persist_directory
#         )
#         print(f"Created new vector database with {vectordb._collection.count()} vectors.")
# except Exception as e:
#     print(f"Error initializing vector database: {e}")
#     raise

# # Use HuggingFaceEmbeddings for both embedding and inference
# from langchain_huggingface import HuggingFaceEndpoint

# llm = HuggingFaceHub(repo_id="google/gemma-1.1-2b-it", model_kwargs={"temperature": 0.7})

# # Define Prompt Template
# template = """
# You are a helpful assistant with access to detailed documents. Your task is to answer the question based on the provided context. If the context does not contain information relevant to the question, you should state that you don't know the answer rather than guessing.

# Use the following context to answer the question at the end. Provide a clear, concise, and accurate response. Your answer should be no longer than three sentences and always end with "Thanks for asking!"

# {context}

# Question:
# {question}

# Answer:
# """
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # Create RetrievalQA Chain
# try:
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectordb.as_retriever(),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )
# except Exception as e:
#     print(f"Error creating QA chain: {e}")
#     raise

# # Solara App
# @solara.component
# def App():
#     query, set_query = solara.use_state("")
#     result, set_result = solara.use_state("")
#     source_documents, set_source_documents = solara.use_state([])

#     def handle_submit():
#         try:
#             response = qa_chain.invoke({"query": query.strip()})
#             set_result(response['result'])
#             set_source_documents(response["source_documents"])
#         except Exception as e:
#             set_result(f"An error occurred: {e}")
#             set_source_documents([])

#     with solara.Card("Document Query Assistant"):
#         solara.Markdown("### Enter your query below:")
#         solara.InputText(label="Your Question", value=query, on_value=set_query)
#         solara.Button("Submit", on_click=handle_submit)

#         if result:
#             solara.Markdown(f"### Answer:\n{result}")

#         if source_documents:
#             solara.Markdown("### Source Documents:")
#             for doc in source_documents:
#                 solara.Markdown(f"- {doc.page_content}")

# # Alias App as Page for Solara CLI
# Page = App



# Updated one and work best
# import os
# from typing import List, Any
# from PyPDF2 import PdfReader, PdfWriter
# import solara
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceHub

# # Set up Hugging Face API token
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# # Merge PDFs
# def merge_pdfs(pdf_directory: str, output_pdf: str) -> None:
#     if not os.path.exists(pdf_directory):
#         raise FileNotFoundError(f"The directory '{pdf_directory}' does not exist.")
#     if os.path.exists(output_pdf):
#         print(f"The merged PDF already exists at {output_pdf}. No need to create it again.")
#         return
#     pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
#     if not pdf_files:
#         raise FileNotFoundError(f"No PDF files found in directory '{pdf_directory}'.")
#     pdf_writer = PdfWriter()
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(pdf_directory, pdf_file)
#         try:
#             pdf_reader = PdfReader(pdf_path)
#             for page in pdf_reader.pages:
#                 pdf_writer.add_page(page)
#         except Exception as e:
#             print(f"Error processing {pdf_file}: {e}")
#     with open(output_pdf, 'wb') as output:
#         pdf_writer.write(output)
#     print(f"Merged PDF saved as {output_pdf}")

# # Directory and merged file path
# pdf_directory = "C:/Users/sauga/Desktop/NLP303/Assessment 3/"
# merged_pdf_path = os.path.join(pdf_directory, 'merged_document.pdf')
# merge_pdfs(pdf_directory, merged_pdf_path)

# # Load Merged PDF and Split into Chunks
# loader = PyPDFLoader(merged_pdf_path)
# pages = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
# splits = text_splitter.split_documents(pages)
# print(f"Number of document chunks created: {len(splits)}")

# # Initialize Vector Store
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# persist_directory = './chroma_db'

# try:
#     if os.path.exists(persist_directory):
#         vectordb = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embedding,
#         )
#         print(f"Using existing vector database with {vectordb._collection.count()} vectors.")
#     else:
#         vectordb = Chroma.from_documents(
#             documents=splits,
#             embedding=embedding,
#             persist_directory=persist_directory
#         )
#         print(f"Created new vector database with {vectordb._collection.count()} vectors.")
# except Exception as e:
#     print(f"Error initializing vector database: {e}")
#     raise

# # Use HuggingFaceHub for inference
# llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5, "max_new_tokens": 250})

# # Improved Prompt Template
# template = """
# You are a knowledgeable and concise assistant. Use the provided context to answer the user's question directly and accurately. Avoid redundancy and unnecessary information.

# ### Context:
# {context}

# ### Question:
# {question}

# ### Response:
# """
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # Create RetrievalQA Chain
# try:
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
#         return_source_documents=False,  # Do not return source documents
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )
# except Exception as e:
#     print(f"Error creating QA chain: {e}")
#     raise

# # Solara App
# @solara.component
# def App():
#     query, set_query = solara.use_state("")
#     result, set_result = solara.use_state("")

#     def handle_submit():
#         try:
#             response = qa_chain.invoke({"query": query.strip()})
#             # Extract and display only the clean answer
#             clean_response = response['result'].strip()
#             set_result(clean_response)
#         except Exception as e:
#             set_result(f"An error occurred: {e}")

#     with solara.Card("Document Query Assistant"):
#         solara.Markdown("### Enter your query below:")
#         solara.InputText(label="Your Question", value=query, on_value=set_query)
#         solara.Button("Submit", on_click=handle_submit)

#         if result:
#             # Display only the answer
#             solara.Markdown(f"**Answer:** {result}")

# # Alias `App` as `Page` for Solara CLI
# Page = App

# Working bot text-to-text


# import os
# from typing import List
# from PyPDF2 import PdfReader, PdfWriter
# import solara
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceHub

# # Set up Hugging Face API token
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# # Merge PDFs
# def merge_pdfs(pdf_directory: str, output_pdf: str) -> None:
#     if not os.path.exists(pdf_directory):
#         raise FileNotFoundError(f"The directory '{pdf_directory}' does not exist.")
#     if os.path.exists(output_pdf):
#         print(f"The merged PDF already exists at {output_pdf}. No need to create it again.")
#         return
#     pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
#     if not pdf_files:
#         raise FileNotFoundError(f"No PDF files found in directory '{pdf_directory}'.")
#     pdf_writer = PdfWriter()
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(pdf_directory, pdf_file)
#         try:
#             pdf_reader = PdfReader(pdf_path)
#             for page in pdf_reader.pages:
#                 pdf_writer.add_page(page)
#         except Exception as e:
#             print(f"Error processing {pdf_file}: {e}")
#     with open(output_pdf, 'wb') as output:
#         pdf_writer.write(output)
#     print(f"Merged PDF saved as {output_pdf}")

# # Directory and merged file path
# pdf_directory = "C:/Users/sauga/Desktop/NLP303/Assessment 3/"
# merged_pdf_path = os.path.join(pdf_directory, 'merged_document.pdf')
# merge_pdfs(pdf_directory, merged_pdf_path)

# # Load Merged PDF and Split into Chunks
# loader = PyPDFLoader(merged_pdf_path)
# pages = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
# splits = text_splitter.split_documents(pages)
# print(f"Number of document chunks created: {len(splits)}")

# # Initialize Vector Store
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# persist_directory = './chroma_db'

# try:
#     if os.path.exists(persist_directory):
#         vectordb = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embedding,
#         )
#         print(f"Using existing vector database with {vectordb._collection.count()} vectors.")
#     else:
#         vectordb = Chroma.from_documents(
#             documents=splits,
#             embedding=embedding,
#             persist_directory=persist_directory
#         )
#         print(f"Created new vector database with {vectordb._collection.count()} vectors.")
# except Exception as e:
#     print(f"Error initializing vector database: {e}")
#     raise

# # Use HuggingFaceHub for inference
# llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5, "max_new_tokens": 250})

# # Improved Prompt Template
# template = """
# You are a knowledgeable and concise assistant. Use the provided context to answer the user's question directly and accurately. Avoid redundancy and unnecessary information.

# ### Context:
# {context}

# ### Question:
# {question}

# ### Response:
# """
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # Create RetrievalQA Chain
# try:
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
#         return_source_documents=False,  # Do not return source documents
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )
# except Exception as e:
#     print(f"Error creating QA chain: {e}")
#     raise

# # Solara App
# @solara.component
# def App():
#     query, set_query = solara.use_state("")
#     chat_history, set_chat_history = solara.use_state([])  # Maintain chat history

#     def handle_submit():
#         try:
#             response = qa_chain.invoke({"query": query.strip()})
#             # Extract only the response part
#             raw_response = response['result'].strip()
            
#             # Process the response to remove unwanted context and instructions
#             # Assuming the response format always includes "Response:" followed by the actual content
#             clean_response = raw_response.split("Response:")[-1].strip()
            
#             # Update chat history with user query and clean bot response
#             set_chat_history(chat_history + [("You", query.strip()), ("Bot", clean_response)])
#             set_query("")  # Clear the input box
#         except Exception as e:
#             set_chat_history(chat_history + [("You", query.strip()), ("Bot", f"An error occurred: {e}")])

#     with solara.Card("Chat with the Assistant"):
#         solara.Markdown("### Chat Interface")
#         # Display chat history
#         for speaker, text in chat_history:
#             solara.Markdown(f"**{speaker}:** {text}")

#         # Input and Submit
#         solara.InputText(label="Your Message", value=query, on_value=set_query)
#         solara.Button("Send", on_click=handle_submit)

# # Alias `App` as `Page` for Solara CLI
# Page = App


# Perfectly workinng code

# import os
# from typing import List
# from PyPDF2 import PdfReader, PdfWriter
# import solara
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceHub

# # Set up Hugging Face API token
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# # Merge PDFs
# def merge_pdfs(pdf_directory: str, output_pdf: str) -> None:
#     if not os.path.exists(pdf_directory):
#         raise FileNotFoundError(f"The directory '{pdf_directory}' does not exist.")
#     if os.path.exists(output_pdf):
#         print(f"The merged PDF already exists at {output_pdf}. No need to create it again.")
#         return
#     pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
#     if not pdf_files:
#         raise FileNotFoundError(f"No PDF files found in directory '{pdf_directory}'.")
#     pdf_writer = PdfWriter()
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(pdf_directory, pdf_file)
#         try:
#             pdf_reader = PdfReader(pdf_path)
#             for page in pdf_reader.pages:
#                 pdf_writer.add_page(page)
#         except Exception as e:
#             print(f"Error processing {pdf_file}: {e}")
#     with open(output_pdf, 'wb') as output:
#         pdf_writer.write(output)
#     print(f"Merged PDF saved as {output_pdf}")

# # Directory and merged file path
# pdf_directory = os.path.join(os.getcwd(), "NLP303/Assessment 3")
# merged_pdf_path = os.path.join(pdf_directory, 'merged_document.pdf')
# merge_pdfs(pdf_directory, merged_pdf_path)

# # Load Merged PDF and Split into Chunks
# loader = PyPDFLoader(merged_pdf_path)
# pages = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
# splits = text_splitter.split_documents(pages)
# print(f"Number of document chunks created: {len(splits)}")

# # Initialize Vector Store
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# persist_directory = './chroma_db'

# try:
#     if os.path.exists(persist_directory):
#         vectordb = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embedding,
#         )
#         print(f"Using existing vector database with {vectordb._collection.count()} vectors.")
#     else:
#         vectordb = Chroma.from_documents(
#             documents=splits,
#             embedding=embedding,
#             persist_directory=persist_directory
#         )
#         print(f"Created new vector database with {vectordb._collection.count()} vectors.")
# except Exception as e:
#     print(f"Error initializing vector database: {e}")
#     raise

# # Use HuggingFaceHub for inference
# llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5, "max_new_tokens": 250})

# # Improved Prompt Template
# template = """
# You are a knowledgeable and concise assistant. Use the provided context to answer the user's question directly and accurately. Avoid redundancy and unnecessary information.

# ### Context:
# {context}

# ### Question:
# {question}

# ### Response:
# """
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # Create RetrievalQA Chain
# try:
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
#         return_source_documents=False,  # Do not return source documents
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )
# except Exception as e:
#     print(f"Error creating QA chain: {e}")
#     raise

# # Solara App
# @solara.component
# def App():
#     query, set_query = solara.use_state("")  # User's current query
#     chat_history, set_chat_history = solara.use_state([])  # Full chat history
    
#     def handle_submit():
#         if query.strip():  # Only process non-empty queries
#             try:
#                 # Retrieve response from the backend
#                 response = qa_chain.invoke({"query": query.strip()})
#                 # Extract the meaningful part of the response
#                 raw_response = response.get('result', '').strip()
#                 clean_response = raw_response.split("Response:")[-1].strip() if "Response:" in raw_response else raw_response

#                  # Update chat history with user query and bot response
#                 set_chat_history(chat_history + [("You", query.strip()), ("Bot", clean_response)])
#                 set_query("")  # Clear input for the next query
#             except Exception as e:
#                 # Handle errors gracefully
#                 set_chat_history(chat_history + [("You", query.strip()), ("Bot", f"An error occurred: {e}")])
#                 set_query("")  # Allow further queries after an error
    
         
#     # Full-Screen Layout
#     with solara.Column(
#         style={
#             "height": "100vh",
#             "width": "100%",
#             "backgroundColor": "#F5F6FA",
#             "display": "flex",
#             "flexDirection": "column",
#             "alignItems": "center",
#             "justifyContent": "center",
#             "padding": "0",
#             "margin": "0",
#         }
#     ):
#         # Header
#         with solara.Card(
#             style={
#                 "width": "100%",
#                 "padding": "15px",
#                 "backgroundColor": "#4A90E2",
#                 "color": "#FFFFFF",
#                 "textAlign": "center",
#                 "fontSize": "24px",
#                 "fontWeight": "bold",
#                 "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.1)",
#             }
#         ):
#             solara.Markdown("💬 Chat Assistant")

#         # Chat History Section
#         with solara.Column(
#             style={
#                 "flexGrow": "1",
#                 "width": "100%",
#                 "padding": "20px",
#                 "overflowY": "auto",  # Enable scrolling for chat history
#                 "backgroundColor": "#FFFFFF",
#                 "scrollBehavior": "smooth",
#             }
#         ):
#             for speaker, text in chat_history:
#                 if speaker == "You":
#                     # User Message
#                     with solara.Row(
#                         style={
#                             "justifyContent": "flex-end",
#                             "marginBottom": "10px",
#                         }
#                     ):
#                         with solara.Card(
#                             style={
#                                 "backgroundColor": "#D0E8FF",
#                                 "borderRadius": "15px",
#                                 "padding": "10px 15px",
#                                 "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
#                                 "maxWidth": "60%",
#                             }
#                         ):
#                             solara.Markdown(f"**{speaker}:** {text}", style={"color": "#004085", "fontSize": "15px"})
#                 else:
#                     # Bot Message
#                     with solara.Row(
#                         style={
#                             "justifyContent": "flex-start",
#                             "marginBottom": "10px",
#                         }
#                     ):
#                         with solara.Card(
#                             style={
#                                 "backgroundColor": "#E3FCEC",
#                                 "borderRadius": "15px",
#                                 "padding": "10px 15px",
#                                 "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
#                                 "maxWidth": "60%",
#                             }
#                         ):
#                             solara.Markdown(f"**{speaker}:** {text}", style={"color": "#225522", "fontSize": "15px"})

#         # Input Section
#         with solara.Row(
#             style={
#                 "width": "100%",
#                 "padding": "15px",
#                 "alignItems": "center",
#                 "backgroundColor": "#F7F8FA",
#                 "borderTop": "1px solid #E5E7EB",
#                 "boxShadow": "0px -2px 4px rgba(0, 0, 0, 0.1)",
#             }
#         ):
#             solara.InputText(
#                 value=query,
#                 on_value=set_query,
#                 label="Type your message...",
#                 style={
#                     "flexGrow": "1",
#                     "padding": "10px",
#                     "border": "1px solid #CCCCCC",
#                     "borderRadius": "20px",
#                     "marginRight": "10px",
#                 },
#             )
#             solara.Button(
#                 "Send",
#                 on_click=handle_submit,
#                 style={
#                     "padding": "10px 20px",
#                     "backgroundColor": "#4CAF50",
#                     "color": "#FFFFFF",
#                     "border": "none",
#                     "borderRadius": "20px",
#                     "cursor": "pointer",
#                     "fontWeight": "bold",
#                     "fontSize": "16px",
#                 },
#             )

# # Alias `App` as `Page` for Solara CLI
# Page = App

# if __name__ == "__main__":
#     import os
#     # Run the Solara app on the correct port and host
#     solara.run(App, port=int(os.environ.get("PORT", 8080)), host="0.0.0.0")

import os
import requests
from typing import List
from PyPDF2 import PdfReader, PdfWriter
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import solara

# Correct GitHub repository raw content URL
GITHUB_RAW_BASE_URL = "https://raw.githubusercontent.com/xaugatp/chatbot/main/"
PDF_FILES = [
    "GEN016.pdf",
    "GEN047DIR.pdf",
    "POL1327DIR.pdf",
    "POL1328DIR.pdf",
    "POL891DIR.pdf",
]

# Download PDFs from GitHub
def download_pdfs(pdf_files: List[str], local_directory: str) -> None:
    """
    Download a list of PDFs from GitHub's raw URLs and save them to a local directory.
    """
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    for file_name in pdf_files:
        pdf_url = GITHUB_RAW_BASE_URL + file_name
        local_path = os.path.join(local_directory, file_name)
        
        if os.path.exists(local_path):
            print(f"{file_name} already exists locally. Skipping download.")
            continue
        
        print(f"Downloading {file_name} from {pdf_url}...")
        
        try:
            response = requests.get(pdf_url, headers=headers, timeout=10)  # Add headers here
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded {file_name} to {local_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_name} from {pdf_url}. Error: {e}")


# Merge PDFs
def merge_pdfs(pdf_directory: str, output_pdf: str) -> None:
    """
    Merge all PDFs in a directory into a single PDF.

    Args:
        pdf_directory (str): Directory containing the PDF files.
        output_pdf (str): Path to save the merged PDF.
    """
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

# Specify the local directory
local_pdf_directory = os.path.join(os.getcwd(), "pdfs")
merged_pdf_path = os.path.join(local_pdf_directory, "merged_document.pdf")

# Step 1: Download PDFs
download_pdfs(PDF_FILES, local_pdf_directory)

# Step 2: Merge PDFs
merge_pdfs(local_pdf_directory, merged_pdf_path)

# Step 3: Load Merged PDF and Split into Chunks
loader = PyPDFLoader(merged_pdf_path)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
splits = text_splitter.split_documents(pages)
print(f"Number of document chunks created: {len(splits)}")

# Step 4: Initialize Vector Store
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

# Step 5: Use HuggingFaceHub for inference
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5, "max_new_tokens": 250})

# Step 6: Improved Prompt Template
template = """
You are a knowledgeable and concise assistant. Use the provided context to answer the user's question directly and accurately. Avoid redundancy and unnecessary information.

### Context:
{context}

### Question:
{question}

### Response:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Step 7: Create RetrievalQA Chain
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
    query, set_query = solara.use_state("")
    chat_history, set_chat_history = solara.use_state([])

    def handle_submit():
        if query.strip():
            try:
                response = qa_chain.invoke({"query": query.strip()})
                result = response.get("result", "").strip()
                set_chat_history(chat_history + [("You", query.strip()), ("Bot", result)])
                set_query("")
            except Exception as e:
                set_chat_history(chat_history + [("You", query.strip()), ("Bot", f"Error: {e}")])
                set_query("")

    with solara.Column():
        solara.InputText(value=query, on_value=set_query)
        solara.Button("Send", on_click=handle_submit)
        for speaker, text in chat_history:
            solara.Markdown(f"**{speaker}:** {text}")

Page = App

if __name__ == "__main__":
    import os
    # Get the port assigned by Render
    port = int(os.environ.get("PORT", 8080))
    # Run the Solara app on the assigned port and listen on all network interfaces
    solara.run(App, port=port, host="0.0.0.0")

