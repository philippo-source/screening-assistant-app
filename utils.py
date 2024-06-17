import time
import openai

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:

        chunks = get_pdf_text(filename)

        docs.append(
            Document(
                page_content=chunks,
                metadata={
                    "name": filename.name,
                    "id": filename.file_id,
                    "type=": filename.type,
                    "size": filename.size,
                    "unique_id": unique_id,
                },
            )
        )
    return docs


def create_embeddings_load_data():
    # embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def push_to_pinecone(
    pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs
):

    # pinecone.init(
    # api_key=pinecone_apikey,
    # environment=pinecone_environment
    # )
    #
    # Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)
    PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)

    index_name = pinecone_index_name
    # PineconeStore is an alias name of Pinecone class, please look at the imports section at the top :)
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index


def pull_from_pinecone(
    pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings
):

    print("20secs delay due to free tier build up...")
    time.sleep(20)
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index


def get_similar_docs(index, query, k, unique_id):

    similar_docs = index.similarity_search_with_score(
        query, int(k), {"unique_id": unique_id}
    )
    return similar_docs


def get_summary(current_doc):
    # llm = OpenAI(temperature=0)
    llm = HuggingFaceHub(
        repo_id="bigscience/bloom", model_kwargs={"temperature": 1e-10}
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
    return summary
