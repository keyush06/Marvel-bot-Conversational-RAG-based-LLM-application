'''
os: Used for interacting with the operating system.
re: Regular expressions for string searching and manipulation.
streamlit as st: The primary framework used to create the web application.
Various imports from langchain_core and langchain_community are used to handle chat history, manage messages, and interact with language models.
Ollama: A specific language model handler (likely from LangChain’s community modules).
'''

import os
import streamlit as st
import rag_chain as rc
import web_scraping as ws
import re

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
import ollama

st.set_page_config(page_title="Chat with a Marvel expert", page_icon="🦜")

# '''
# chunk_type, chunk_size, and chunk_overlap are used to define how documents should be split into smaller parts (chunks) for processing.
# SESSION_ID: A constant session identifier for tracking conversations in this example.
# These tags (B_INST, E_INST, B_SYS, E_SYS) are used to format instructions and system messages in a structured way. They are placeholders for LLM instructions and system prompts.
# '''

chunk_type = "Recursive"
chunk_size = 800
chunk_overlap = 100

SESSION_ID = "1234"
B_INST, E_INST = "<s>[INST]", "[/INST]</s>"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def create_context_aware_chain(retriever, model_name):
    '''This function creates a context-aware chain for reformulating questions based on previous chat history. The reformulated questions are standalone and can be understood without additional context.'''
    llm_summarise = Ollama(model=model_name, temperature=0.0, num_predict=256)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is.")

    hist_aware_retr = rc.create_rephrase_history_chain(llm_summarise, retriever, contextualize_q_system_prompt)

    return hist_aware_retr


def create_answering_chain(model_name, retriever):
    '''Creates a RAG chain for answering questions about Marvel Universe. This chain uses a language model and document retriever to generate concise answers.'''
    llm_answer = Ollama(model=model_name, temperature=0.5, num_predict=256)

    system_prompt = (
        "You are an expert in Marvel movies, characters and actors who answers questions of new fans of the Marvel Cinematic Universe. "
        "You will be given some information extracted from Wikipedia that can be useful to answer the question, use them how you want. "
        "If you don't know the answer, say 'The information given are not enough to answer to the question'. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "ADDITIONAL INFORMATION: {context}"
    )


    full_rag_chain_with_history = rc.create_qa_RAG_chain_history(llm_answer, retriever, system_prompt)

    return full_rag_chain_with_history


def chunk_docs(docs):
    '''This function splits documents into smaller chunks for processing. It first splits by HTML headers and then recursively by text separators.'''
    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"), ("h4", "Header 4")]
    params = {}

    match chunk_type:
        case "Recursive":
            separators = ["\n\n", "\n", "(?<=\. )", " ", ""]
            params["separators"] = separators
            params["chunk_size"] = chunk_size
            params["chunk_overlap"] = chunk_overlap

            chunks = rc.chunking(docs, headers_to_split_on, chunk_type, **params)

    print("Number of chunks: ", len(chunks))
    print("Number of documents: ", len(docs))

    return chunks


def create_retriever(directory):

    '''Creates a document retriever by processing documents, chunking them, creating a vector store, and returning a retriever object.
        Document Loading: Documents are loaded from the specified directory.
        Chunking and Vector Store: Documents are chunked and stored as vectors in an index.
        Retriever: A retriever is created based on cosine similarity, retrieving the top 5 most similar chunks.
    '''

    print("Importing Documents...")
    docs = rc.read_docs(directory)

    print("Creating chunks")
    chunks = chunk_docs(docs)

    print("Creating a vector store")
    embedding_model, vector_index = rc.create_vector_index_and_embedding_model(chunks)
    print("Generating retriever")
    retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return retriever


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    '''Session Management: Chat history is stored in st.session_state.chat_store. If no history exists for the session, a new ChatMessageHistory object is created.
        Return: The function returns the chat history object for the current session.'''
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

# '''
# In the next method, there are two main things to consider: -
# Invocation: The full chain is invoked with the user’s query and session configuration.
# Answer Extraction: The resulting answer is extracted from the chain's output.
# '''

def get_response(full_chain, session_id, user_query):
    answer = full_chain.invoke({"input": user_query}, config={
        "configurable": {"session_id": session_id}
    })

    return answer["answer"]

def folder_added():
    '''This function checks if a folder has been added to the app and creates a retriever object based on the documents in the folder.'''
    with st.spinner("Creating vector retriever, please wait..."):
        st.session_state.disabled_sum_model = False
        st.session_state.retriever = create_retriever(st.session_state.files_folder)


# '''
#     These functions update the selected summarization and answering models and create the necessary chains.
#     Spinner: Displays a loading spinner while the retriever is being created.
#     Session State Update: Enables the summarization model dropdown and saves the retriever in session state.
# '''


# Summarization Model: The summarization model is updated and the retriever chain is created.
# Answering Model: The answering model is updated and the final chain is created, which includes session history management.

def update_summarization_model():
    st.session_state.disabled_ans_model = False
    model_name = st.session_state.sum_model

    ollama_model_name = re.search("(.*)  Size:", model_name).group(1)
    st.session_state.retriever_chain = create_context_aware_chain(st.session_state.retriever, ollama_model_name)

def update_answer_model():
    model_name = st.session_state.ans_model
    ollama_model_name = re.search("(.*)  Size:", model_name).group(1)

    st.session_state.retriever_answer_chain = create_answering_chain(ollama_model_name, st.session_state.retriever_chain)

    st.session_state.final_chain = RunnableWithMessageHistory(
        st.session_state.retriever_answer_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


#-------------------------------------------------#
### Streamlit App Layout ###
#-------------------------------------------------#

# st.set_page_config(page_title="Chat with a Marvel expert", page_icon="🦜")
st.title("_Chat_ with :blue[A Marvel Expert] 🤖")

if "disabled_ans_model" not in st.session_state:
    st.session_state.disabled_ans_model = True

if "disabled_sum_model" not in st.session_state:
    st.session_state.disabled_sum_model = True

with st.sidebar:
    models_ollama = ollama.list()["models"]
    model_name = [m['name'] for m in models_ollama]
    model_size = [float(m["size"]) for m in models_ollama]
    name_detail = zip(model_name, model_size)
    name_detail = sorted(name_detail, key=lambda x: x[1])
    model_info = [f"{name}  Size: {size/(10**9):.2f}GB" for name, size in name_detail]
    st.text_input("Insert the folder containing the .html files to be used for RAG", on_change=folder_added, key="files_folder")
    st.selectbox("Choose a model for context summarization", model_info, index=None, on_change=update_summarization_model, placeholder="Select model", key="sum_model", disabled=st.session_state.disabled_sum_model)
    st.selectbox("Choose a model for answering", model_info, index=None, on_change=update_answer_model, placeholder="Select model", key="ans_model", disabled=st.session_state.disabled_ans_model)
    st.caption("You will see here the models downloaded from Ollama. For installing ollama: https://ollama.com/download.  \nFor the models available: https://ollama.com/library.  \n⚠️ Remember: Heavily quantized models will perform slightly worse but much faster.")




if "disabled" not in st.session_state:
    st.session_state.disabled = False

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

def disable():
    st.session_state.disabled = True

def enable():
    st.session_state.disabled = False

# Create the history
if SESSION_ID not in st.session_state.chat_store:
    st.session_state.chat_store[SESSION_ID] = ChatMessageHistory()


# prints the chat history going through the chat store that is created
for message in st.session_state.chat_store[SESSION_ID].messages:
    MESSAGE_TYPE = "AI" if isinstance(message, AIMessage) else "Human"
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "Human"

    with st.chat_message(MESSAGE_TYPE):
        st.write(message.content)

## User input handling##

if user_query := st.chat_input("Type your message ✍", disabled=st.session_state.disabled, on_submit=disable):
    if "retriever" not in st.session_state:
        st.error("Retriever, summarization model and answering model were not set.")
    elif "retriever_chain" not in st.session_state:
        st.error("Summarization model and answering model were not set.")
    elif "retriever_answer_chain" not in st.session_state:
        st.error("Answering model was not set.")
    elif user_query is not None and user_query != "":
        with st.spinner("The model is generating an answer, please wait"):
            response = get_response(st.session_state.final_chain, SESSION_ID, user_query)
            st.session_state.disabled = False
            st.rerun()






