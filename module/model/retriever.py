from module.model.datamodel import *
from  module.model.customFaiss import faiss_engine
import pandas as pd
        
class retriever():
    def __init__(self, document:pd.DataFrame, method = faiss_engine):
        """Retriever instance

        Args:
            document (pd.DataFrame): Whole document
            method (str): Defaults to 'faiss'
        """
        self.engine = method(document["core_question"].to_list())
        self.document_storage = DocumentStore(metaData=document, retrieverEngine=self.engine)
        
    def get_content(self, query_index: int) -> ResponseCandidate:
        
        df = self.document_storage.metaData
        row = df[df['index'] == query_index]
        
        if row.empty:
            raise ValueError(f"No row found with index {query_index}")
        
        return ResponseCandidate(**row.iloc[0].to_dict())
        
    def retrieve(self, question, k=3):
        
        result = self.engine.retrieve(question, k)
        
        return [self.get_content(res) for res in result]