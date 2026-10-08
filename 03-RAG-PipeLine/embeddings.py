from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeEmbeddings

# Our Pinecone index was created with dimension 1536, but multilingual-e5-large
# returns 1024-dim vectors. Padding with zeros keeps cosine similarity identical.
INDEX_DIMENSION = 1536


class PaddedPineconeEmbeddings(Embeddings):
    def __init__(self, model: str = "multilingual-e5-large", dimension: int = INDEX_DIMENSION):
        self.base = PineconeEmbeddings(model=model)
        self.dimension = dimension

    def _pad(self, vector: list[float]) -> list[float]:
        return vector + [0.0] * (self.dimension - len(vector))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._pad(v) for v in self.base.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._pad(self.base.embed_query(text))
