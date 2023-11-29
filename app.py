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
        num_results=10,
        score_threshold=0.4,
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
        self.num_results = num_results
        self.score_threshold = score_threshold
        self._load_database()
        self._process_embeddings()

    def _process_text_chunks(self):
        self.text_chunks = self.file_loader.load_and_split(text_splitter=self.text_splitter)

    def _load_database(self):
        # attempt to load vector database from disk
        db_loaded_from_disk = Chroma(
            embedding_function=self.embedding_fn,  # embedding function to use
            collection_metadata={
                "hnsw:space": "cosine"
            },  # needed to keep similarity scores from being negative
            persist_directory="./chroma_db",  # where to store the vector store on disk
        )

        if db_loaded_from_disk._collection.count() > 0:
            self.chroma_db = db_loaded_from_disk
            print("Loaded vector database from disk")
            return

        self._process_text_chunks()

        # save text chunks into a vector store that persists on disk
        self.chroma_db = Chroma.from_documents(
            self.text_chunks,  # text chunks to load
            self.embedding_fn,  # embedding function to use
            collection_metadata={
                "hnsw:space": "cosine"
            },  # needed to keep similarity scores from being negative
            persist_directory="./chroma_db",  # where to store the vector store on disk
        )

    def _process_embeddings(self):
        # create a retriever for the vector store
        self.retriever = self.chroma_db.as_retriever(
            search_type="similarity_score_threshold",  # use a score threshold to filter results
            search_kwargs={
                "k": self.num_results,  # number of results to return
                "score_threshold": self.score_threshold,  # only returns documents above this threshold
            },
        )

        # create a chain that combines the retriever and the language model
        self.qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
        )

        # embedding = self.embedding_fn.embed_documents("Example text to turn into vector embeddings")
        # print(f"len(embedding[0])={len(embedding[0])}")
        # print(f"embedding[0][:10]={embedding[0][:10]}")

    def ask(self, query):
        return self.qa_with_sources(query)

    def _debug(self):
        print("========= DEBUG =========")

        if hasattr(self, "text_chunks"):
            print(f"Text chunks: len(text_chunks)={len(self.text_chunks)}")
        print(
            f"Collections in db: chroma_db._collection.count()={self.chroma_db._collection.count()}"
        )

        query = input("Enter query: ")

        print(self.ask(query))

        docs = self.retriever.get_relevant_documents(query)
        print(f"search results against retriever (max={self.num_results}): len(docs)={len(docs)}")
        print(f"retriever score_threshold={self.score_threshold}")
        user_input = input("Show docs? (y/n): ")
        if user_input.lower() == "y":
            pprint(docs)

        user_input = input(
            "Show docs from alt search method against db (includes similarity score, think max is 4)? (y/n): "
        )
        if user_input.lower() == "y":
            db_docs = self.chroma_db.similarity_search_with_score(query)
            pprint(db_docs)
