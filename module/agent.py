from module.model.gemini import Gemini

class helpdeskAgent():
    def __init__(self, retriever):
        self.retriver_engine = retriever
        self.agent = Gemini()
        
    def retrieve(self, question):
        
        return self.retriver_engine.retrieve(question)
            
    def response(self, question):
        
        candidates = self.retrieve(question)
        prompt = f"""
        Using context below,
        
        Context: {candidates[0].to_dict()["core_answer"]}

        Answer this question: {question}?
        """
        print([can.to_qa() for can in candidates])
        
        return self.agent.generate(prompt)