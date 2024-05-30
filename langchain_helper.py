from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint


 

load_dotenv()
HF_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')


model_name = "distilbert/distilbert-base-cased-distilled-squad"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedddings = HuggingFaceBgeEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )



def crate_vector_db_from_youtube(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)

    docs = text_splitter.split_documents(transcript)

    db =FAISS.from_documents(docs , embedding=embedddings)

    return db



def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=512, temperature=0.5, token=HF_TOKEN
    )

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain({'question' : query, 'docs' : docs_page_content})
    # print(response['text'])
    # response = response.replace("\n", "")
    return response['text'], docs


# vector_store = crate_vector_db_from_youtube(video_url)

# get_response_from_query(vector_store, "What is the title of this video?")
