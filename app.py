import sys
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import shutil
from langchain.schema import AIMessage  



# Enable debugging logs
import logging
logging.basicConfig(level=logging.DEBUG)

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")
os.environ["OPENAI_API_KEY"] = api_key

# Load the text files from the rag_data folder
rag_data_dir = "./rag_data/"
sys.path.append(rag_data_dir)

# Initialize document loaders
loaders = []
for filename in os.listdir(rag_data_dir):
    file_path = os.path.join(rag_data_dir, filename)
    try:
        loaders.append(TextLoader(file_path, encoding="utf-8"))
        logging.debug(f"Loaded file: {filename}")
    except Exception as e:
        st.warning(f"Failed to load: {filename}. Error: {e}")

# Load and merge documents
data = []
for loader in loaders:
    try:
        data.extend(loader.load())
    except Exception as e:
        st.warning(f"Error loading data from {loader}: {e}")

# Exit if no data is loaded
if not data:
    st.error("No documents could be loaded from the specified directory. Please check your data files.")
    sys.exit()
else:
    # st.write(f"Loaded {len(data)} documents successfully.")
    logging.debug(f"Total documents loaded: {len(data)}")

# Merge the documents
merged_documents = [Document(page_content=" ".join([doc.page_content for doc in data]))]

# Initialize embedding model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Rebuild the vectorstore if needed
persist_directory = "./chroma_db"  # Directory to store the Chroma index
shutil.rmtree(persist_directory, ignore_errors=True)  # Clear the directory for a fresh rebuild
os.makedirs(persist_directory, exist_ok=True)  # Ensure the directory exists
vectorstore = Chroma.from_documents(merged_documents, embeddings, persist_directory=persist_directory)
# st.write("Vectorstore rebuilt successfully.")

# Custom RAG prompt template
template = """
You are a teaching assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question at the end.
The following pieces of retrieved context are from a Network Security textbook.
The name of the textbook is Computer Security: Principles and Practice, 4th Edition.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't say anything mean or offensive.

Context: {context}

Question: {question}
"""
custom_rag_prompt = ChatPromptTemplate.from_template(template)

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Specify the model
    temperature=0.2,  # Control the creativity of the response
    max_tokens=4000
)

# Initialize the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}  # Retrieve the top 5 matches
)

# Define the chain (prompt and LLM)
rag_chain = custom_rag_prompt | llm


def get_response(query):
    try:
        # Log the query
        logging.debug(f"Query: {query}")

        # Retrieve documents
        docs = retriever.get_relevant_documents(query)
        logging.debug(f"Retrieved documents: {len(docs)}")

        if not docs:
            return "I'm sorry, I couldn't find any relevant context for your query."

        # Prepare context from retrieved documents
        context = " ".join([doc.page_content for doc in docs])

        # Format the input for the chain
        inputs = {"context": context, "question": query}

        # Pass inputs to the chain
        response = rag_chain.invoke(inputs)

        # Assuming the response is an AIMessage object, access its 'content' attribute directly
        if isinstance(response, AIMessage):  # Ensure the response is an instance of AIMessage
            response_text = response.content
        else:
            # If response is not an AIMessage, convert to string
            response_text = str(response)

        logging.debug(f"Response text: {response_text}")

        return response_text
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"Error processing your request: {e}"



# # Function to handle RAG chain response
# def get_response(query):
#     try:
#         # Log the query
#         st.write(f"Processing query: {query}")
#         logging.debug(f"Query: {query}")

#         # Retrieve documents
#         docs = retriever.get_relevant_documents(query)
#         st.write(f"Retrieved {len(docs)} documents for the query.")
#         logging.debug(f"Retrieved documents: {len(docs)}")

#         if not docs:
#             return "I'm sorry, I couldn't find any relevant context for your query."

#         # Prepare context from retrieved documents
#         context = " ".join([doc.page_content for doc in docs])
#         st.write(f"Retrieved context: {context[:500]}...")  # Show part of the context for debugging

#         # Format the input for the chain
#         inputs = {"context": context, "question": query}

#         # Pass inputs to the chain
#         response = rag_chain.invoke(inputs)
#         logging.debug(f"Response: {response}")
#         return response
#     except Exception as e:
#         logging.error(f"Error processing query: {e}")
#         return f"Error processing your request: {e}"

################################################################## Streamlit interface
st.title("Ask a Question About Network Security")
st.markdown("### Enter your question below to get answers")

# # Example query for debugging
# example_query = "How can I prevent a DOS attack?"
# st.write(f"Example query: {example_query}")
# response = get_response(example_query)
# st.write(f"Example response: {response}")

# User input question
query = st.text_input("Enter your question:")

# Button to submit the query
if st.button("Submit"):
    if query.strip():
        # Get and display the response
        response = get_response(query)
        if response.startswith("Error"):
            st.error(response)
        else:
            st.success(f"**Answer:** {response}")
    else:
        st.warning("Please enter a question before submitting.")
