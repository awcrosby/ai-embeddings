import os

from pprint import pprint

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()


class RAGHandler:
    def __init__(
        self,
        chunk_size=400,
        chunk_overlap=100,
        temperature=0.2,
        max_tokens=200,
        model_name="gpt-3.5-turbo",
        file_path="harvard-healthy-living-guide.pdf",
    ):
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.file_loader = PyPDFLoader(file_path=file_path)
        self.embedding_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatOpenAI(
            openai_api_key=self.OPENAI_API_KEY,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._process_text_chunks()
        self._process_embedding()

    def _process_text_chunks(self):
        self.text_chunks = self.file_loader.load_and_split(text_splitter=self.text_splitter)

    def _process_embedding(self):
        self.chroma_db = Chroma.from_documents(
            self.text_chunks, self.embedding_fn, collection_metadata={"hnsw:space": "cosine"}
        )
        kwargs = {
            "k": 10,  # number of results to return
            "score_threshold": 0.2,  # only returns documents with a score above that threshold
        }
        self.retriever = self.chroma_db.as_retriever(
            search_type="similarity_score_threshold", search_kwargs=kwargs
        )
        self.qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
        )

    def ask(self, query):
        return self.qa_with_sources(query)

    def _debug(self, query="this is some text"):
        print("========= DEBUG =========")
        embedding = self.embedding_fn.embed_documents(query)
        print(f"query={query}")
        if embedding:
            print(f"len(embedding[0])={len(embedding[0])}")
            print(f"embedding[0][:10]={embedding[0][:10]}")

        print(f"len(text_chunks)={len(self.text_chunks)}")
        print(f"chroma_db._collection.count()={self.chroma_db._collection.count()}")

        print("vector db search results:")
        docs = self.retriever.get_relevant_documents(query)
        print(f"len(docs)={len(docs)}")
        for doc in docs:
            pprint(doc.page_content.replace("\n", ""))
