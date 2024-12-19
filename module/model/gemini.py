import requests
import re
import json
import streamlit as st

from typing import Any, List

class TokenManager:
    @staticmethod
    def fetch_token(credential: str, url: str = 'http://10.9.93.83:8443/google-authen') -> str:
        """Fetch an authentication token using the provided credential."""
        headers = {'Content-Type': 'application/json'}
        data={'service_account': credential}
        response = requests.post(url, data, headers)
        response.raise_for_status()
        return response.json()['Token']

    @staticmethod
    def get_local_credential(filepath: str) -> str:
        """Retrieve credential from a local file and fetch the token."""
        with open(filepath, 'r') as file:
            credential = file.read()
        return TokenManager.fetch_token(credential)
    
    @staticmethod
    def get_streamlit_credential() -> str:
        credential = json.loads(st.secrets["kong_cred"])
        return TokenManager.fetch_token(credential)
        

    @staticmethod
    def get_token(location: str, local_cred_path: str = "/Users/sirabhobs/Desktop/poc-hr-helpdesk-chatbot/credential/embd_cred.json") -> str:
        """Retrieve a token based on the specified location."""
        if location == 'local':
            return TokenManager.get_local_credential(local_cred_path)
        elif location == 'streamlit':
            return TokenManager.get_streamlit_credential()
        else:
            raise ValueError("Invalid location specified. Choose 'local' or 'server'.")
        
class Gemini:
    def __init__(self):
        self.emb_url = "http://10.9.93.83:8443/contactcenter-textembedding"
        self.gemini_url = "http://10.9.93.83:8443/contactcenter-gemini1_5-flash"
        self.max_output = 8192
        self.temperature = 0

        self.token = TokenManager.get_token('local')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of documents."""
        return [self._call_embedding(text) for text in texts]

    def embed(self, text: str) -> List[float]:
        """Compute an embedding for a single query."""
        return self._call_embedding(text)

    def generate(self, prompt: str, 
                 system_instruction:str = "You are a female helpful HR helpdesk assistant for Krungthai Bank. Your role is to form an answer that you'll be given and reply back in Thai", 
                 is_json_output: bool = False):
        """Synchronous call to Gemini API"""
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "systemInstruction": {
                "parts": [
                {
                    "text": system_instruction
                }
            ]
            },
            "generation_config": {
                "maxOutputTokens": self.max_output,
                "temperature": self.temperature,
                "seed": 42,
                "responseMimeType": "application/json" if is_json_output else "text/plain",
            },
        }

        response = requests.post(self.gemini_url, json=data, headers=headers)
        response.raise_for_status()
        return self.process_json_response(response.json()) if is_json_output else self.process_response(response.json())

    def _call_embedding(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = {
            "instances": [{"content": text}],
            "parameters": {"autoTruncate": True},
        }
        response = requests.post(self.emb_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['predictions'][0]['embeddings']['values']
    
    def process_response(self, result: Any):
        """Processes the API response and logs information"""
        candidates = [
            candidate.get("content", {}).get("parts", [])[0].get("text", "")
            for line in result if "candidates" in line
            for candidate in line["candidates"]
        ]

        return "".join(filter(None, candidates))
    
    def process_json_response(self, result: Any):
        
        json_string = self.process_response(result)
        
        try:
            cleaned = re.sub(r"```[a-zA-Z]*\n|```$", "", json_string.strip(), flags=re.MULTILINE)
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("Invalid JSON format detected.")
            cleaned = cleaned[json_start:json_end]
            output = json.loads(cleaned)
            return output
        except:
            return "Cannot parse to json format"