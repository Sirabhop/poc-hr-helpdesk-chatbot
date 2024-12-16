from module.model.gemini import Gemini

class helpdeskAgent():
    def __init__(self, retriever):
        self.retriver_engine = retriever
        self.agent = Gemini()
        self.memory = []
        
    def get_memory(self):
        system_prompt = """
        The following is the memory of the HelpdeskAgent containing previous interactions and context. 
        Use this information to generate accurate and contextually appropriate responses.
        """
        conversation = "\n".join([f"Employee: {item['Employee']}\nAI: {item['AI']}" for item in self.memory])
        return system_prompt + "\n" + conversation

    def update_meory(self, q, a):
        self.memory.append({'Employee':q, 'AI':a})
        
    def get_system_instruction(self):
        
        if self.get_memory():
            memory_propt = f"""
            Prvoding with the conversation history:
            {self.get_memory()}
            """
            
        system_instruction = f"""
        You are a female helpful HR helpdesk assistant for Krungthai Bank. 
        Your role is to form an answer that you'll be given and reply back in Thai.
        Without prying and asking for private information.
        {memory_propt}
        """
        
        return system_instruction
        
    def retrieve(self, question):
        print('AI is retrieving relevant information...')

        return self.retriver_engine.retrieve(question)
                
    def thinking(self, question):
        print('AI is thinking...')
        header = f"""
        Given a following question from Krungthai banking employee: {question}.
        Do you think you can guess what the employee is asking about?
        If yes return next_step_flag as "next_step_rag" and answers with your guess or a paraphrase of employee question
        Otherwise, return next_step_flag "next_step_clarify" and create the follow_up_question to either recheck or clarify the question.
        
        """
        output_format = """
        Output format is in json format:
        
        ```json
        {
            'next_step_flag': 'either next_step_rag or next_step_clarify',
            'follow_up_question': 'either NA or the follow up question (if the question is not clear and you decided that next_step_clarify)'
        }
        ```
        """
        
        decision_prompt = header + output_format
    
        return self.agent.generate(decision_prompt, self.get_system_instruction(), is_json_output=True)
        
    def response(self, question): 
        
        thinking_result = self.thinking(question)
        next_step_flag = thinking_result.get("next_step_flag")
        follow_up_question = thinking_result.get("follow_up_question")
        
        if next_step_flag == "next_step_clarify":
            
            self.update_meory(question, follow_up_question)
            
            return follow_up_question
        
        elif next_step_flag == "next_step_rag":
            
            candidates = self.retrieve(question)
            
            prompt = f"""
            Using context below,
            
            Context: {candidates[0].to_dict()["core_answer"]}

            Answer this question: {question}?
            """
            
            answer = self.agent.generate(prompt, self.get_system_instruction())
            
            self.update_meory(question, answer)
        
            return answer