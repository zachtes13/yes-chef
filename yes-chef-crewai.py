import streamlit as st
import pdfplumber as pdf_p
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.header("Yes Chef!")
with st.sidebar:
    st.title("Home Cookbook")
    files = st.file_uploader("Upload your own recipe PDFs!", type="pdf", accept_multiple_files=True)

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

if files:
    all_chunks = []
    
    for file in files:
        file_id = f"{file.name}_{file.size}"
        
        if file_id not in st.session_state.processed_files:
            with pdf_p.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"

            text_splitter = RecursiveCharacterTextSplitter(
                separators = ["\n\n", "\n", ". ", " ", ""],
                chunk_size = 500,
                chunk_overlap = 100
            )
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            
            st.session_state.processed_files.add(file_id)
    
    if all_chunks:
        embeddings = OpenAIEmbeddings(
            model = "text-embedding-3-small",
            openai_api_key = OPENAI_API_KEY
        )
        
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISS.from_texts(all_chunks, embeddings)
        else:
            new_vector_store = FAISS.from_texts(all_chunks, embeddings)
            st.session_state.vector_store.merge_from(new_vector_store)

with st.form(key="menu_form", clear_on_submit=False):
    user_prompt = st.text_input("What's on the menu?", key="user_prompt")
    submitted = st.form_submit_button("Send")

if submitted and user_prompt:
    if st.session_state.vector_store is not None:
        with st.spinner("Retrieving relevant recipes..."):
            retriever = st.session_state.vector_store.as_retriever(
                search_type = "mmr",
                search_kwargs = { "k": 4 }
            )
            
            relevant_docs = retriever.invoke(user_prompt)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        with st.spinner("Chefs are brainstorming..."):
            recipe_researcher = Agent(
                role='Recipe Researcher',
                goal='Analyze recipe context and extract relevant information',
                backstory="""You are an expert culinary researcher with deep knowledge of 
                international cuisines, cooking techniques, and recipe development. You excel 
                at analyzing recipe documents and identifying the most relevant information.""",
                verbose=True,
                allow_delegation=False
            )

            seasonal_chef = Agent(
                role='Seasonal Recipe Researcher',
                goal='Provide supplementary meal suggestions on how to incorporate seasonal ingredients to make seasonal dishes',
                backstory="""You are a professional chef who prioritizes menus using seasonal
                ingredients. Your emphasis on seasonally accessible ingredients ensures freshness and culinary quality.""",
                verbose=True,
                allow_delegation=False
            )

            dietary_chef = Agent(
                role='Dietary Recipe Researcher',
                goal='Provide supplementary meal suggestions on how to incorporate nutritional ingredients to make healthy dishes',
                backstory="""You are a professional chef who prioritizes healthy food and
                recipes that are nutritionally balanced and prioritize a healthy lifestyle.""",
                verbose=True,
                allow_delegation=False
            )
            
            chef_writer = Agent(
                role='Recipe Writer',
                goal='Write clear, detailed, and engaging culinary responses',
                backstory="""You are a professional recipe writer who creates easy-to-follow 
                recipes with precise measurements and clear instructions. You make cooking 
                accessible and enjoyable for home cooks.""",
                verbose=True,
                allow_delegation=False
            )
            
            research_task = Task(
                description=f"""Analyze the following recipe context and user question:
                
                USER QUESTION: {user_prompt}
                
                RECIPE CONTEXT FROM UPLOADED PDFs:
                {context}
                
                Your task: Extract the most relevant information from the context that answers 
                the user's question. Identify key recipes, ingredients, techniques, or cooking 
                tips that are pertinent. If the context doesn't fully answer the question, 
                note what's missing.""",
                agent=recipe_researcher,
                expected_output="A summary of relevant information from the recipe context"
            )

            seasonal_task = Task(
                description=f"""Using the research findings, supplement the written response with 3-5 bullet point ideas 
                on how to incorporate seasonal ingredients in a response to the user's question: "{user_prompt}"
                
                You should only be adding short, bullet point, recommendations and tips regarding seasonal cooking.
                Append to the response with a sub-header and bullet points.""",
                agent=seasonal_chef,
                context=[research_task],
                expected_output="A relevant header to seasonal cooking with 3-5 bullet point suggestions"
            )

            dietary_task = Task(
                description=f"""Using the research findings, supplement the written response and seasonal suggestions 
                with 3-5 bullet point ideas on how to incorporate nutritionally balanced
                ingredients in a response to the user's question: "{user_prompt}"
                
                You should only be adding short, bullet point, recommendations and tips regarding healthy cooking.
                Append to the response and seasonal suggestion with a sub-header and bullet points.""",
                agent=dietary_chef,
                context=[research_task, seasonal_task],
                expected_output="A relevant header to healthy cooking with 3-5 bullet point suggestions"
            )

            writing_task = Task(
                description=f"""Using the research findings, create a comprehensive response 
                to the user's question: "{user_prompt}"
                
                Format your response to be:
                - Clear and easy to understand
                - Well-structured with proper formatting
                - Practical and actionable
                - Based on the provided recipe context
                - Warm and enthusiastic in tone
                
                If the context contains specific recipes, include ingredients and instructions.
                If the context is insufficient, supplement with general culinary knowledge 
                while being clear about what came from the uploaded recipes vs general knowledge.

                IMPORTANT: At the very end of your response, append the following two sections in this order:
                1) A seasonal cooking section that includes the seasonal_task output.
                2) A healthy cooking section that includes the dietary_task output.

                Preserve the section headers and bullet points from those task outputs (copy them verbatim when possible).""",
                agent=chef_writer,
                context=[research_task, seasonal_task, dietary_task],
                expected_output="A complete, well-formatted response to the user's question"
            )
            
            crew = Crew(
                agents=[recipe_researcher, chef_writer, seasonal_chef, dietary_chef],
                tasks=[research_task, seasonal_task, dietary_task, writing_task],
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
        
        st.success("Order up!")
        st.markdown("---")
        st.markdown(result)
    else:
        st.info("Without uploaded recipes, the chefs will work with general culinary knowledge only.")

        with st.spinner("Chefs are brainstorming..."):
            recipe_researcher = Agent(
                role='Recipe Researcher',
                goal='Analyze recipe context and extract relevant information',
                backstory="""You are an expert culinary researcher with deep knowledge of 
                        international cuisines, cooking techniques, and recipe development. You excel 
                        at analyzing recipe documents and identifying the most relevant information.""",
                verbose=True,
                allow_delegation=False
            )

            seasonal_chef = Agent(
                role='Seasonal Recipe Researcher',
                goal='Provide supplementary meal suggestions on how to incorporate seasonal ingredients to make seasonal dishes',
                backstory="""You are a professional chef who prioritizes menus using seasonal
                        ingredients. Your emphasis on seasonally accessible ingredients ensures freshness and culinary quality.""",
                verbose=True,
                allow_delegation=False
            )

            dietary_chef = Agent(
                role='Dietary Recipe Researcher',
                goal='Provide supplementary meal suggestions on how to incorporate nutritional ingredients to make healthy dishes',
                backstory="""You are a professional chef who prioritizes healthy food and
                        recipes that are nutritionally balanced and prioritize a healthy lifestyle.""",
                verbose=True,
                allow_delegation=False
            )

            chef_writer = Agent(
                role='Recipe Writer',
                goal='Write clear, detailed, and engaging culinary responses',
                backstory="""You are a professional recipe writer who creates easy-to-follow 
                        recipes with precise measurements and clear instructions. You make cooking 
                        accessible and enjoyable for home cooks.""",
                verbose=True,
                allow_delegation=False
            )

            research_task = Task(
                description=f"""Analyze the following user question:
                
                USER QUESTION: {user_prompt}
                
                Your task: Extract the most relevant information from the context that answers 
                the user's question. Identify key recipes, ingredients, techniques, or cooking 
                tips that are pertinent. If the context doesn't fully answer the question, 
                note what's missing.""",
                agent=recipe_researcher,
                expected_output="A summary of relevant information from the recipe context"
            )

            seasonal_task = Task(
                description=f"""Using the research findings, supplement the written response with 3-5 bullet point ideas 
                        on how to incorporate seasonal ingredients in a response to the user's question: "{user_prompt}"
                        
                        You should only be adding short, bullet point, recommendations and tips regarding seasonal cooking.
                        Append to the response with a sub-header and bullet points.""",
                agent=seasonal_chef,
                context=[research_task],
                expected_output="A relevant header to seasonal cooking with 3-5 bullet point suggestions"
            )

            dietary_task = Task(
                description=f"""Using the research findings, supplement the written response and seasonal suggestion
                            with 3-5 bullet point ideas on how to incorporate nutritionally balanced ingredients 
                            in a response to the user's question: "{user_prompt}"
                            
                            You should only be adding short, bullet point, recommendations and tips regarding healthy cooking.
                            Append to the seasonal suggestion and response with a sub-header and bullet points.""",
                agent=dietary_chef,
                context=[research_task, seasonal_task],
                expected_output="A relevant header to healthy cooking with 3-5 bullet point suggestions"
            )

            writing_task = Task(
                description=f"""Using the research findings, create a comprehensive response 
                        to the user's question: "{user_prompt}"
                        
                        Format your response to be:
                        - Clear and easy to understand
                        - Well-structured with proper formatting
                        - Practical and actionable
                        - Warm and enthusiastic in tone

                        IMPORTANT: At the very end of your response, append the following two sections in this order:
                        1) A seasonal cooking section that includes the seasonal_task output.
                        2) A healthy cooking section that includes the dietary_task output.

                        Preserve the section headers and bullet points from those task outputs (copy them verbatim when possible).""",
                agent=chef_writer,
                context=[research_task, seasonal_task, dietary_task],
                expected_output="A complete, well-formatted response to the user's question"
            )

            crew = Crew(
                agents=[recipe_researcher, chef_writer, seasonal_chef, dietary_chef],
                tasks=[research_task, seasonal_task, dietary_task, writing_task],
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
        
        st.success("Order up!")
        st.markdown("---")
        st.markdown(result)
