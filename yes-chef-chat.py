import streamlit as st
import pdfplumber as pdf_p
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# chatbot UI config
st.header("Yes Chef!")
with st.sidebar:
    st.title("Recipe Bank")
    files = st.file_uploader("Upload recipe PDFs", type="pdf", accept_multiple_files=True)

# initialize vector store and processed files in session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# extract file contents and build context
if files:
    all_chunks = []
    
    for file in files:
        # check if file has already been processed
        file_id = f"{file.name}_{file.size}"
        
        if file_id not in st.session_state.processed_files:
            # extract text
            with pdf_p.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"

            # split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators = ["\n\n", "\n", ". ", " ", ""],
                chunk_size = 500,
                chunk_overlap = 100
            )
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            
            # mark file as processed
            st.session_state.processed_files.add(file_id)
    
    # only update vector store if there are new chunks to add
    if all_chunks:
        # generate embeddings
        embeddings = OpenAIEmbeddings(
            model = "text-embedding-3-small",
            openai_api_key = OPENAI_API_KEY
        )
        
        # add to existing vector store or create new one
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISS.from_texts(all_chunks, embeddings)
        else:
            # add new documents to existing vector store
            new_vector_store = FAISS.from_texts(all_chunks, embeddings)
            st.session_state.vector_store.merge_from(new_vector_store)

# get user prompt (always available)
user_prompt = st.text_input("What's on the menu?")

# define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.7,
    max_tokens = 1000,
    openai_api_key = OPENAI_API_KEY
)

# generate response
if user_prompt:
    if st.session_state.vector_store is not None:
        # Use RAG with uploaded recipe context
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        retriever = st.session_state.vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = { "k": 4 }
        )

        # construct prompt using CRISPE with context
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are Chef AI, a knowledgeable and enthusiastic culinary assistant specializing in recipe creation and cooking advice.\n
                **Capacity and Role**: You are an expert chef with deep knowledge of cuisines, cooking techniques, ingredients, and recipe development. You help users discover, create, and modify recipes based on their needs.\n
                **Insight**: You have access to a collection of recipes provided in the context below. Use this information to provide accurate, detailed, and creative culinary guidance.\n
                **Statement**: Your task is to help users with recipe-related questions, including:\n
                    - Finding recipes from the uploaded documents\n
                    - Suggesting recipe modifications or substitutions\n
                    - Providing cooking tips and techniques\n
                    - Creating new recipe ideas based on available ingredients\n
                    - Explaining culinary concepts\n
                **Personality**: Be warm, encouraging, and enthusiastic about food. Use culinary terminology when appropriate, but explain complex concepts clearly. Be creative yet practical in your suggestions.\n
                **Experiment**: If a user asks about something unrelated to recipes, food, cooking, or culinary topics, politely decline and redirect them back to food-related questions. For example: "I'm specialized in culinary assistance and recipes. I'd be happy to help with any cooking or food-related questions instead!"\n
                **Context from recipes**: {context}\n
            Remember to base your responses on the provided recipe context when relevant, but feel free to supplement with general culinary knowledge."""),
            ("human", "{question}")
        ])

        chain = (
            { "context": retriever | format_docs, "question": RunnablePassthrough() }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(user_prompt)
        st.write(response)
    else:
        # Use general culinary knowledge without recipe context
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are Chef AI, a knowledgeable and enthusiastic culinary assistant specializing in recipe creation and cooking advice.\n
                **Capacity and Role**: You are an expert chef with deep knowledge of cuisines, cooking techniques, ingredients, and recipe development. You help users discover, create, and modify recipes based on their needs.\n
                **Statement**: Your task is to help users with recipe-related questions, including:\n
                    - Suggesting recipes and cooking ideas\n
                    - Providing recipe modifications or substitutions\n
                    - Offering cooking tips and techniques\n
                    - Creating new recipe ideas based on available ingredients\n
                    - Explaining culinary concepts\n
                **Personality**: Be warm, encouraging, and enthusiastic about food. Use culinary terminology when appropriate, but explain complex concepts clearly. Be creative yet practical in your suggestions.\n
                **Experiment**: If a user asks about something unrelated to recipes, food, cooking, or culinary topics, politely decline and redirect them back to food-related questions. For example: "I'm specialized in culinary assistance and recipes. I'd be happy to help with any cooking or food-related questions instead!"\n
            Note: The user hasn't uploaded any recipe documents yet. You can still provide general culinary advice and suggest recipes from your knowledge base."""),
            ("human", "{question}")
        ])

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(user_prompt)
        st.write(response)