from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# Memory setup
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Key required by ConversationalRetrievalChain
    return_messages=True
)

def build_qa_chain(docs):
    # Create FAISS vector store
    db = FAISS.from_documents(docs, embeddings)

    # Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Create Conversational QA chain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    return qa_chain
