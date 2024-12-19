import faiss
import numpy as np
import streamlit as st
from module.model.gemini import Gemini

class faiss_engine:
    def __init__(self, document, index_type="flat_l2", nlist=100, pq_m=8, hnsw_m=32):
        """
        Initialize Custom FAISS engine.
        :param document: List of documents or text to build embeddings.
        :param index_type: FAISS index type ('flat_l2', 'flat_ip', 'ivf', 'pq', 'ivf_pq', 'hnsw').
        :param nlist: Number of clusters for IVF-based methods.
        :param pq_m: Number of sub-vectors for Product Quantization.
        :param hnsw_m: Number of edges per node for HNSW index.
        """
        self.index_type = index_type
        self.nlist = nlist
        self.pq_m = pq_m
        self.hnsw_m = hnsw_m
        self.embedder = Gemini()  # Embedding model
        
        # Generate embeddings
        embeddings = self.__embed_document(document)
        # Build the FAISS engine
        self.engine = build_faiss_index(embeddings, index_type, nlist, pq_m, hnsw_m)
        st.write("Building FAISS instance...")

    def __embed_document(self, document):
        """Generate embeddings from document."""
        # return np.array(self.embedder.embed_documents(document)).astype('float32')
        return np.load("./module/model/exported_embeddings.npy")

    def save_index(self, index_file_path):
        """Save the FAISS index to a file."""
        faiss.write_index(self.engine, index_file_path)
        print(f"Index saved to {index_file_path}")

    def load_index(self, index_file_path):
        """Load a FAISS index from a file."""
        self.engine = faiss.read_index(index_file_path)
        print(f"Index loaded from {index_file_path}")

    def retrieve(self, query: list, k: int = 5):
        """Retrieve top-k similar results for a query."""
        emb_queries = np.array([self.embedder.embed(q) for q in query], dtype="float32")
        distances, indices = self.engine.search(emb_queries, k)
        return [{'index': a, 'distance': b} for a, b in zip(indices[0], distances[0])]


def build_faiss_index(embeddings, index_type="flat_l2", nlist=100, pq_m=8, hnsw_m=32):
    """
    Build a FAISS index based on the given parameters.
    :param embeddings: NumPy array of embeddings to add to the index.
    :param index_type: FAISS index type ('flat_l2', 'flat_ip', 'ivf', 'pq', 'ivf_pq', 'hnsw').
    :param nlist: Number of clusters for IVF-based methods.
    :param pq_m: Number of sub-vectors for Product Quantization.
    :param hnsw_m: Number of edges per node for HNSW index.
    :return: FAISS index.
    """
    st.write("Building FAISS instance...")
    dimension = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"Dimension: {dimension}, Index Type: {index_type}")
    
    if index_type == "flat_l2":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "flat_ip":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
    elif index_type == "pq":
        index = faiss.IndexPQ(dimension, pq_m, 8)  # nbits=8
        index.train(embeddings)
    elif index_type == "ivf_pq":
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, pq_m, 8)
        index.train(embeddings)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, hnsw_m)
    else:
        raise ValueError("Unsupported index type")
    
    # Add embeddings to the index
    index.add(embeddings)
    print(f"FAISS engine built with {index_type}. Total vectors: {index.ntotal}")
    return index