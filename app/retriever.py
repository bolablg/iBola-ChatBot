from langchain_chroma import Chroma

import config
from utils.embedder import get_embeddings


def get_retriever():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.DB_PATH, embedding_function=embeddings
    )
    # Using MMR search to balance relevance and diversity. Fetch 20 docs and select the top 8.
    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20}
    )
