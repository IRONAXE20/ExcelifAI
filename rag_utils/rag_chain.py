from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.utils import DistanceStrategy
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                   encode_kwargs={"normalize_embeddings": True}
                                  )

llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

def build_qa_chain(docs):
    db = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    return qa_chain
