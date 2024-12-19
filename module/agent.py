from module.model.gemini import Gemini

class helpdeskAgent():
    def __init__(self, retriver_engine, preference):
        self.retriver_engine = retriver_engine
        self.agent = Gemini()
        self.preference = preference
        self.memory = []
        
    def get_memory(self):
        system_prompt = """
        The following is the memory of the HelpdeskAgent containing previous interactions and context. 
        Use this information to generate accurate and contextually appropriate responses.
        """
        
        conversation = "\n".join([f"Employee: {item['Employee']}\nAI: {item['AI']}" for item in self.memory])
        return system_prompt + "\n" + conversation

    def update_memory(self, q, a):
        self.memory.append({'Employee':q, 'AI':a})
        
    def check_hallucination(self, q, a, doc):
        print('AI is checking hallucination...')
        instruction = f"""
        You are a grader assessing whether an LLM generation is grounded in and supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in and supported by the set of facts.
        """
        
        prompt = f"""Set of facts: \n {doc} \n LLM generation: {a}"""
        
        return True if self.agent.generate(prompt, instruction) == 'yes' else False
    
    def check_answer(self, q, a):
        print('AI is checking answer...')
        instruction = f"""
        You are a grader assessing whether an answer addresses and resolves a question \n 
        Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.
        """
        
        prompt = f"""
        User question: \n {q} \n LLM generation: {a}
        """
        
        return True if self.agent.generate(prompt, instruction) == 'yes' else False
        
    def check_relevant_doc(self, q, doc):
        print('AI is checking relevant doc...')
        instruction = f"""
        You are a grader assessing relevance of a retrieved document to a user question.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        """
        
        prompt = f"""Retrieved document: \n\n {doc} \n\n User question: {q}"""
        
        return True if self.agent.generate(prompt, instruction) == 'yes' else False
    
    def get_system_instruction(self):
        
        if self.get_memory():
            memory_propt = f"""
            Prvoding with the conversation history:
            {self.get_memory()}
            """
            
        system_instruction = f"""
        You are a female helpful HR helpdesk assistant for Krungthai Bank. 
        Your role is to form an answer that you'll be given and reply back in Thai.
        
        And always response in the following manner {self.preference}
        
        Without prying and asking for private information.
        {memory_propt}
        """
        
        return system_instruction
        
    def retrieve(self, q):
        print(f'AI is retrieving relevant information for {q}...')

        return self.retriver_engine.retrieve([q])
                
    def thinking(self, q):
        print('AI is thinking...')
        header = f"""
        Given a following question from Krungthai banking employee: {q}.
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
    
    def response(self, q, candidates):
        context = candidates[0].to_dict()["core_answer"]
        print(f"AI is responding with context: Q: {candidates[0].to_dict()["core_question"]} \n A: {candidates[0].to_dict()["core_answer"]}")
        prompt = f"""
        Using ONLY and ALL the context below to answer the following question:
        Context: {context}
        Answer this question: {q}?
        """
        return self.agent.generate(prompt, self.get_system_instruction())
        
    def run(self, question):
        print("--- Helpdesk Agent Flow Start ---")

        # Step 1: Thinking Node
        thinking_result = self.thinking(question)
        next_step_flag = thinking_result.get("next_step_flag")
        follow_up_question = thinking_result.get("follow_up_question")
        print(f'AI is done thinking: {next_step_flag} and {follow_up_question}')
        
        if next_step_flag == "next_step_clarify":
            self.update_memory(question, follow_up_question)
            return follow_up_question

        # Step 2: Retrieval Node
        relevant_candidates = self.retrieve(question)
        
        # relevant_candidates = []
        # for doc in candidates:
        #     if self.check_relevant_doc(question, doc.to_dict()["core_answer"]):
        #         relevant_candidates.append(doc)

        # if not relevant_candidates:
        #     return "ฉันไม่พบข้อมูลที่เกี่ยวข้อง รบกวนถามใหม่อีกครั้งนะคะ"

        # Step 3: Response Node
        response = self.response(question, relevant_candidates)
        
        # Step 4: Hallucination Check Node
        is_hallucinated = self.check_hallucination(question, response, relevant_candidates)
        # Step 5: Answer Check Node
        is_answer_correct = self.check_answer(question, response)
        
        # Step 6: Update Memory Node
        if not (is_hallucinated & is_answer_correct):
            self.update_memory(question, response)
            return response