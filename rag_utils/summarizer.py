from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    "Summarize the following content in simple language:\n\n{input_text}"
)

chain = prompt | llm

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def summarize_text(text):
    chunks = split_text(text)
    summaries = [chain.invoke({"input_text": chunk}).content for chunk in chunks]
    return "\n\n".join(summaries)
