from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# TODO
prompt = "Are there any Pearl Jam lyrics about stars?"

# Note: we must use the same embedding model that we used when uploading the docs
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Querying the vector database for "relevant" docs
document_vectorstore = PineconeVectorStore(index_name="pearl-jam-ten-lyrics", embedding=embeddings)
retriever = document_vectorstore.as_retriever()
context = retriever.get_relevant_documents(prompt)
for doc in context:
    print(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n")
print("__________________________")

# Adding context to our prompt
template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
prompt_with_context = template.invoke({"query": prompt, "context": context})

# Asking the LLM for a response from our prompt with the provided context
llm = ChatOpenAI(temperature=0.3)
results = llm.invoke(prompt_with_context)

print(results.content)