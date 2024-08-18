from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredHTMLLoader

'''
HTMLSectionSplitter: Splits HTML documents based on specific HTML tags (like headers) for easier processing.
RecursiveCharacterTextSplitter: Recursively splits text into smaller chunks based on specific delimiters.
TextLoader, DirectoryLoader, UnstructuredHTMLLoader: Loaders to read documents from directories, which can handle various formats like plain text or HTML.
'''

from langchain.embeddings import CacheBackedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore

'''
CacheBackedEmbeddings: Caches the embeddings for faster retrieval. It helps during the inference time when you have to retrieve embeddings for a large number of documents.
HuggingFaceEmbeddings: Uses HuggingFace models to create embeddings from text.
FAISS: A library to perform fast similarity searches on vectors (used to retrieve relevant documents).
LocalFileStore: Stores cached data locally.
'''

import torch
import transformers
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface.llms import HuggingFacePipeline


'''
Now we create the methods for the retrieval chain and the document combination chain. We also create the chat prompt template and the callback handler. Thes era ethe methods that 
will handle the HTML files, the embeddings, and the retrieval of the relevant documents.
'''

def read_docs(folder_path):

    '''This function reads all .html files in a specified folder, loading them into a list of documents.'''

    loader = DirectoryLoader(folder_path, glob="*.html", show_progress=True, use_multithreading=False, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    return docs

def chunking(docs, headers, chunk_type = 'Recursive',**kwargs):

    '''Splits documents into smaller chunks, first by HTML headers and then recursively by text separators. This is useful for processing large documents by breaking them into more manageable pieces.
'''

    html_splitter = HTMLSectionSplitter(headers)
    html_header_splits = html_splitter.split_documents(docs)

    match chunk_type:
        case "Recursive":
            split_doc = recursive_split(splits=html_header_splits, **kwargs)
        case _:
            split_doc = recursive_split(splits=html_header_splits, **kwargs)

    return split_doc


def recursive_split(splits, separators=("\n\n", "\n", "(?<=\. )", " ", ""), chunk_size=800, chunk_overlap=100):

    ''''This function further splits documents using specified text separators and chunk size, adding some overlap between chunks. The overlap ensures context continuity between chunks.
'''
    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    recursive_header_split = rec_char_splitter.split_documents(splits)
    return recursive_header_split


def create_vector_index_and_embedding_model(chunks):

    '''Creates an embedding model using HuggingFace and stores document embeddings in a FAISS vector index for efficient retrieval. The embeddings are cached to speed up subsequent queries.
'''
    store = LocalFileStore("./cache/")

    embed_model_id = 'intfloat/e5-small-v2'
    model_kwargs = {"device": "cpu", "trust_remote_code": True}

    embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)

    embedder = CacheBackedEmbeddings.from_bytes_store(embeddings_model, store, namespace=embed_model_id)

    #CREATE VECTOR INDEX
    vector_index = FAISS.from_documents(chunks, embedder)

    return embeddings_model, vector_index


def create_qa_RAG_chain_history(llm_pipeline, retriever, system_prompt):
    '''Creates a Retrieval-Augmented Generation (RAG) chain that includes chat history for conversational continuity. The chain retrieves relevant documents and uses them to generate responses.
'''
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                  MessagesPlaceholder("chat_history"),
                                                  ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(llm_pipeline, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def create_rephrase_history_chain(llm_pipeline, retriever, system_prompt):
    '''This function creates a retriever that is aware of the chat history, helping to contextualize queries based on past interactions. This is crucial for maintaining a coherent conversation.'''
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                               MessagesPlaceholder("chat_history"),
                                                               ("human", "{input}")])

    history_aware_retriever = create_history_aware_retriever(llm_pipeline, retriever, contextualize_q_prompt)

    return history_aware_retriever

def answer_LLM_only(model, tokenizer, query):
    '''Uses a pre-trained language model to generate answers directly from the model's knowledge, without document retrieval. This is useful for simple queries where external documents aren't necessary.'''
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    query_tokenized = tokenizer.encode_plus(query, return_tensors="pt")["input_ids"].to('cuda')
    answer_ids = model.generate(query_tokenized,
                                max_new_tokens=256,
                                do_sample=True)

    decoded_answer = tokenizer.batch_decode(answer_ids)

    return decoded_answer

def retrieve_top_k_docs(query, vector_index, embedding_model, k=5):
    '''Retrieves the top k most relevant documents from the vector index based on the similarity to the query's embedding.'''
    query_embedding = embedding_model.embed_query(query)
    docs = vector_index.similarity_search_by_vector(query_embedding, k=k)

    return docs


def generate_model(model_id):
    '''This function loads a language model from HuggingFace, applies 4-bit quantization to reduce memory usage, and prepares it for inference. This is essential for running large models on hardware with limited resources.
'''
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, config=model_config,
                                                              quantization_config=bnb_config, device_map="auto")

    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def create_pipeline(model, tokenizer, temperature, repetition_penalty, max_new_tokens):
    '''Creates a pipeline for generating text using a pre-trained language model. The pipeline allows for easy text generation with customizable parameters like temperature and repetition penalty.
    This function creates a text generation pipeline using the specified model and tokenizer, with parameters like temperature to control the creativity of the model's outputs.'''
    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=pipeline)

    return llm_pipeline

'''Just an extra method below. Creates a simpler RAG chain without chat history, suitable for QA tasks where the context of previous interactions isn't as important.'''
def create_qa_RAG_chain(llm_pipeline, retriever, system_prompt):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                               ("human", "{input}")])

    qa_chain = create_stuff_documents_chain(llm_pipeline, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)

    return chain



