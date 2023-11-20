import os
import openai

import pypdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate 
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
import shutil



from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


llm_model = "gpt-3.5-turbo-0301" # llm model



def create_vector_space():

    # load document
    
    file = "utils/document.pdf"
    docs = PyPDFLoader(file)
    docs = docs.load()
    
    # create a text splitter instance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )

    splits = text_splitter.split_documents(docs) # splits the docs into chunks

    embedding = OpenAIEmbeddings()


    # path to database/vectorstores
    persist_directory = 'utils/chroma/'

    try:
        os.mkdir("utils/chroma")
    except:
        shutil.rmtree("utils/chroma")
        os.mkdir("utils/chroma")




    # generate embeddings for our document
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )


    vectordb.persist() # save embeddings


def build_prompt(question):

    create_vector_space()

    # Load document embeddings from a vectorstores db
    persist_directory = 'utils/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)



    # Build prompt

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer try make up the up an answer as accurate as possible. If there is code enclose it in triple backtics. Always say "I am an experiment, my answers may be inaccurate at the end.
    {context}
    Question: {question}
    Helpful Answer:"""




    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,) # prompt template




    llm = ChatOpenAI(model_name=llm_model, temperature=0)  # load llm model



    # create instance for memory


    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # create a retrieval chain for Q&A


    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type = "stuff",
        retriever=retriever,
        memory=memory
    )

    result = qa({"question": question})

    return result["answer"]



        # # Question 1
        # question = "Title of the context given to you"
        # result = qa({"question": question})

        # display(Markdown(result["answer"]))



        # # Question 1

        # question = "Give a brief summary of the context"
        # result = qa({"question": question})



        # # Question 1
        # question = "Using the context, summarise lambda in simple terms please."
        # result = qa({"question": question})

        # display(Markdown(result["answer"]))

