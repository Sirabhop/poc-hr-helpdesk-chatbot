from  module.model.gemini import Gemini
import faiss
import numpy as np

class faiss_engine():
    
    def __init__(self, document):
        self.embedder = Gemini()
        self.engine = self.build(document)
        
    def __embed_document(self, document):
        return np.array(self.embedder.embed_documents(document)).astype('float32')
        
    def build(self, document):
        
        embedding_matrix = self.__embed_document(document)
        retriver_engine = faiss.IndexFlatL2(embedding_matrix.shape[1])
        retriver_engine.add(embedding_matrix)
        
        return retriver_engine
    
    def import_faiss(self, index_file_path):
        retriver_engine = faiss.read_index(index_file_path)
        
        return retriver_engine
    
    def retrieve(self, question, k):
        
        emb_question = self.embedder.embed(question)
        emb_question = np.array([emb_question], dtype='float32')  #query
        result = self.engine.search(emb_question, k)
        
        return result[1][0]
