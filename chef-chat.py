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
    st.title("Recipes")
    file = st.file_uploader("Upload a recipe PDF!", type="pdf")

# extract file contents
if file is not None:
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

    # generate embeddings
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        openai_api_key = OPENAI_API_KEY
    )

    # store embeddings in vector DB
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user prompt
    user_prompt = st.text_input("What's on the menu?")

    # generate response
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = { "k": 4 }
    )

    # define LLM
    llm = ChatOpenAI(
        model = "gpt-4o-mini",
        temperature = 0.7,
        max_tokens = 1000,
        openai_api_key = OPENAI_API_KEY
    )

    # construct prompt using CRISPE
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

    if user_prompt:
        response = chain.invoke(user_prompt)
        st.write(response)