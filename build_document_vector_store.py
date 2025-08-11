from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import glob
from itertools import chain
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    separators=['\n\n', '\n']
)

# 특정 폴더 내 모든 .md 파일 경로 가져오기
markdown_files = glob.glob("./documents/*.md")

# 각 파일을 로드하고 text_splitter로 분할
loaders = [UnstructuredMarkdownLoader(path) for path in markdown_files]
document_list = list(chain.from_iterable(
    loader.load_and_split(text_splitter) for loader in loaders
))


embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

vector_store = Chroma.from_documents(
    documents=document_list,
    embedding=embeddings,
    collection_name = 'document_collection',
    persist_directory = './document_collection'
)
vector_store.persist()