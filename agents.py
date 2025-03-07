from openai import OpenAI
import os
import re

class Agent():
    def __init__(self):
        BASE_URL = "https://litellm.ml.scaleinternal.com/"
        self.client = OpenAI(
            api_key=os.getenv("litellm_key"),
            base_url=BASE_URL,
            timeout=600.0, 
            max_retries=1
        )
        self.INTERNAL_USER = "internal"
        self.cost_tags = {
            "metadata": {
                "tags": ["useCase:LLMasJudge"]
            }
            }
        
class Respondent(Agent):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def get_response(self, question):
        llm_output = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            user=self.INTERNAL_USER,
            extra_body=self.cost_tags,
        )
        response = (llm_output.choices[0].message.content)
        return response
    
class Judge(Agent):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.RUBRIC = " 1. Instruction Following: All prompt instructions and constraints are fully satisfied by the response. 2. Relevance: All information in the response is required by the prompt or adds helpful and enriching information relevant to the prompt. There is no entirely irrelevant information. 3. Helpfulness: The response contains sufficient detail and specificity to satisfy all parts of the prompt. 4. Accuracy of Claims: All facts that are objectively verifiable are correct, aligning with established facts or evidence. 5. No Misleading Information: All claims are appropriately contextualized where necessary, and opinions/controversial statements are not stated as fact. 6. No Hallucinations: All people, events, websites, articles, or institutions referenced in the response are real (not hallucinated). 7. Numbers and Calculations: All equations, dates, figures, and other numerical information is correct and presented accurately (with wording to indicate approximation if appropriate, and correct syntax). 8. Readability: The response is clear and easy to read. 9. Language Mechanics - The response has no grammar issues, typos, syntax errors, etc 10. Outline and Coherence: The order and presentation of ideas are coherent and have a good flow. 11. Concision- The response efficiently communicate concepts and key points without being verbose."
        self.SCORING_PROMPT = f"You are an impartial judge. You will receive a prompt and a response. Rate the response in the context of the prompt using this evaluation criteria: {self.RUBRIC} Assign 1 point for the response meeting each of the criteria, and 0 for not meeting them. Provide a brief justification of your scoring. Compute the overall score as a sum of individual scores. Return the Overall Score followed by the justification."
        self.PREFERENCE_PROMPT =  f"You are an impartial judge. You will receive a prompt and a set of responses makred as Response 1, Response 2, and so on. Choose the best response in the context of the prompt using this evaluation criteria: {self.RUBRIC}. Only return the best response number. For eg. if you think Response 2 is better than Response 1, only reply the number 2."
    
    def _extract_score(self, response_text):
        score, reason = None, None
        # Look for a number after "Overall Score:" which might have spaces in between.
        pattern = r'Overall Score\s*[:\-]?\s*(\d+)'
        match = re.search(pattern, response_text)
        if match:
            # Convert the captured group to an integer
            score = int(match.group(1))
            reason = response_text[match.end():].strip()
            reason = re.sub(r'^/11\s*', '', reason) 
        return score, reason
    
    def _get_llm_output(self, prompt):
        try:
            llm_output = self. client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            user=self.INTERNAL_USER,
            extra_body=self.cost_tags,
            )
            llm_output = llm_output.choices[0].message.content
            return llm_output
        except Exception as e:
            print("Error during API call")
            return None
    
    def _extract_preference(self, response, num_choices):
        preference, feedback = None, None
        match = re.search(r'\b([0-9])\b', response)
        # print('match:',match.group(1), type(match.group(1)))
        if match:
            preference_idx = int(match.group(1))
            # print('pref idx:', preference_idx, 'num choices:', num_choices)
            if preference_idx <= num_choices:
                preference = preference_idx - 1
                # print('pref:', preference)
                return preference, feedback
        print('Invalid Response Number. Trying again.')
        feedback = f"Invalid Response. Try Again. Make sure to pick only one of the available response numbers. Return the number only. "+self.PREFERENCE_PROMPT
        return preference, feedback

    def get_score(self, question, response):
        original_prompt=f"""{self.SCORING_PROMPT}
            Question:
            {question}

            Response:
            {response}
          """
        judge_prompt = original_prompt
        tries = 0
        while True:
            response = self._get_llm_output(judge_prompt)
            if response is not None:
                # print('llm response:', response)
                score, reason = self._extract_score(response)
                if score is not None and reason is not None:
                    return score, reason
                print('Invalid Score:', score, reason)
                feedback = f"Invalid Score. Try Again. Make sure to the overall score is an integer. "+self.SCORING_PROMPT
                judge_prompt = feedback + original_prompt # to avoid accumulating feedback
                tries += 1
                if tries >=3:
                    break
    
    def get_preference(self, question, answers):
        original_prompt=f"""{self.PREFERENCE_PROMPT}
                Question:
                {question}"""
        for idx, ans in enumerate(answers):
            original_prompt = original_prompt+f"""
                Response {idx+1}:
                {ans}"""
        judge_prompt = original_prompt
        tries = 0
        while True:    
            llm_output = self._get_llm_output(judge_prompt)
            if llm_output is not None:
                # print('llm out:', llm_output)
                preference, feedback = self._extract_preference(llm_output, len(answers))
                # print('preference:', preference)
                if preference is not None:
                    return preference
                judge_prompt = feedback + original_prompt
                tries += 1
                if tries >= 3:
                    break

        