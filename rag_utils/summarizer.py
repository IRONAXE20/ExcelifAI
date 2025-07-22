from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    "Summarize the following content in simple language:\n\n{input_text}"
)

# ✅ Create the chain using LCEL-style composition
chain = prompt | llm

# Split long text into manageable chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Generate summaries for each chunk
def summarize_text(text):
    chunks = split_text(text)
    summaries = [chain.invoke({"input_text": chunk}).content for chunk in chunks]
    return "\n\n".join(summaries)
