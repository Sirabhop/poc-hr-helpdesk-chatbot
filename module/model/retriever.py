from module.model.datamodel import *
from  module.model.customFaiss import faiss_engine
import pandas as pd
import streamlit as st

class retriever():
    def __init__(self, document:pd.DataFrame):
        """Retriever instance

        Args:
            document (pd.DataFrame): Whole document
            method (str): Defaults to 'faiss'
        """
        st.write("Building Retriver instance...")
        self.engine = faiss_engine(document)
        self.document_storage = DocumentStore(metaData=document, retrieverEngine=self.engine)        
        
    def get_content(self, queries: int) -> ResponseCandidate:
        
        index = int(queries['index'])
        distance = queries['distance']

        dict_row = self.document_storage.metaData.iloc[index].to_dict()
        dict_row['distance'] = distance
        
        return ResponseCandidate(**dict_row)
        
    def retrieve(self, question, k=3):
        
        result = self.engine.retrieve(question, k)
        
        return [self.get_content(res) for res in result]