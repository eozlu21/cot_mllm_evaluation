import json
import random
from typing import List, Dict
from datasets import load_dataset
from cot_mllm_evaluation.mllm.huggingface import HuggingFaceMLLM
from cot_mllm_evaluation.verifier.huggingface import LLMVerifier
import re
# Four rethinking strategies
search_strategies = [
    ("Backtracking", """<question>
{}
</question>
<|image|>
<previous reasoning>
{}
</previous reasoning>

<response requirements>
You must revise your previous reasoning attempt by identifying an **incorrect assumption** made during the analysis and tracing how it affected the final conclusion.

Your Chain of Thought must follow:
1. **Verification**: Clearly state the flawed assumption that led to the incorrect conclusion.
2. **Inner Thinking**: Build a new reasoning chain that corrects that assumption.
3. **Final Conclusion**: Present the revised conclusion.
4. **Verification**: Assess if the new reasoning is more valid.

Your Final Conclusion must directly follow from the revised Inner Thinking steps. Do not reuse or rephrase prior conclusions unless your new logic genuinely supports them.

Output format:
```json
{{
"CoT": [
    {{"action": "Verification", "content": "Flawed assumption: [state it]"}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""),

    ("Exploring New Paths", """<question>
{}
</question>
<|image|>
<previous reasoning>
{}
</previous reasoning>

<response requirements>
You must now pursue a **completely different interpretation** of what makes the cartoon uncanny.

1. Do not reuse any reasoning points or visual elements from the prior attempt.
2. Explore elements or relationships that were previously ignored.
3. Avoid repeating sentence structures or vocabulary.
4. Your **Final Conclusion** must directly follow from the Inner Thinking steps in this response.
5. Your **Verification** must confirm that the Final Conclusion is clearly supported by those steps. If not, revise the reasoning.

Your Final Conclusion must directly follow from the revised Inner Thinking steps. Do not reuse or rephrase prior conclusions unless your new logic genuinely supports them.


Respond in strict JSON format:
```json
{{
"CoT": [
    {{"action": "Verification", "content": "Exploring new reasoning path"}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""),

    ("Verification", """<question>
{}
</question>
<|image|>
<previous reasoning>
{}
</previous reasoning>

<response requirements>
1. First, identify whether the Final Conclusion accurately describes the uncanniness of the cartoon image.
2. If incorrect, choose **one generic reason** why the reasoning is flawed from the list below:

- "Missed key contradiction in the scene"
- "Focused on a normal element, not the uncanny one"
- "Misinterpreted an unusual object or behavior"
- "Reasoning lacks support from visual cues"
- "Uncanny factor mentioned, but not clearly explained"
- "Multiple uncanny elements, but reasoning fixated on the wrong one"

3. Then start a new reasoning attempt based on that flaw.
   - Your **Inner Thinking** must take a different path than the prior reasoning.
   - Your **Final Conclusion** must be directly supported by the revised Inner Thinking steps and must fix the earlier mistake without introducing new ones.
   - Your **Verification** step must explain why the conclusion logically follows from the current reasoning.


Your Final Conclusion must directly follow from the revised Inner Thinking steps. Do not reuse or rephrase prior conclusions unless your new logic genuinely supports them.

Output in this exact JSON format:

```json
{{
    "CoT": [
        {{"action": "Verification", "content": "Reasoning flaw: [insert selected category here]"}},
        {{"action": "Inner Thinking", "title": "...", "content": "..."}},
        ...
        {{"action": "Final Conclusion", "content": "..."}},
        {{"action": "Verification", "content": "..."}}
    ]}}```"""),

    ("Correction", """<question>
{}
</question>
<|image|>
<previous reasoning>
{}
</previous reasoning>

<response requirements>
Identify the specific flaw in the previous reasoning (e.g., a faulty logical step or unsupported observation), then revise **only** the affected parts while preserving sound logic.

Format:
1. **Verification**: Identify which part of the reasoning (by title or content) was flawed.
2. **Inner Thinking**: Rewrite the reasoning to correct that specific error.
3. **Final Conclusion**: Provide the corrected conclusion.
4. **Verification**: Assess if the revised reasoning is now well supported.

Your Final Conclusion must directly follow from the revised Inner Thinking steps. Do not reuse or rephrase prior conclusions unless your new logic genuinely supports them.

The Verification must confirm that the conclusion is now fully supported by the revised reasoning path.

Output format:
```json
{{
"CoT": [
    {{"action": "Verification", "content": "Correction target: [step title or reasoning type]"}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""),
    
    ("Persona Shift", """<question>
{}
</question>
<|image|>
<previous reasoning>
{}
</previous reasoning>

<response requirements>
Now view the cartoon through the eyes of a **long-time New Yorker magazine reader**. Consider the kind of subtle absurdities, cultural critiques, or ironic twists typical of New Yorker humor.


1. **Inner Thinking**: Interpret the visual scene through the lens of layered social meaning, contradiction, or dry humor.
2. **Final Conclusion**: Suggest what may feel uncanny to such a reader.
3. **Verification**: Evaluate whether this perspective uncovers a new dimension of uncanniness.

Your Final Conclusion must emerge logically from the Inner Thinking steps using this reader persona.

Your Final Conclusion must directly follow from the revised Inner Thinking steps. Do not reuse or rephrase prior conclusions unless your new logic genuinely supports them.

Your Verification must confirm that the conclusion is fully supported by your persona-based interpretation.

Output in strict JSON format:
```json
{{
"CoT": [
    {{"action": "Verification", "content": "Shifting to a New Yorker reader perspective"}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```""")]


class CoTEvaluator:
    def __init__(self, dataset_name: str, mllm: HuggingFaceMLLM, verifier: LLMVerifier, fewshot: List[Dict] = None, explanation_type: str = "uncanny"):
        self.dataset_name = dataset_name
        self.mllm = mllm
        self.verifier = verifier
        #self.fewshot = fewshot or []
        self.stats = {"correct": 0, "total": 0}
        self.results = []
        self.explanation_type = explanation_type
        
        
    def conclusion_is_reused(self, new_conclusion: str, previous_conclusions: List[str]) -> bool:
        # Normalize text to check semantic repetition
        def normalize(text: str) -> str:
            return re.sub(r'\W+', ' ', text.lower()).strip()

        norm_new = normalize(new_conclusion)
        for prev in previous_conclusions:
            norm_prev = normalize(prev)
            if norm_prev in norm_new or norm_new in norm_prev:
                return True
        return False

        
    def parse_cot_json(self, response: str) -> List[Dict]:
        try:
            idx = response.rfind(")")
            response = response[:idx] + "}" + response[idx+1:]
            if not response.strip().startswith("{"):
                response = response[response.find("{"):response.rfind("}")+1]
            response += "```"
            #print(f"RESPONSE: {response}")
            
            
            json_blocks = re.findall(r"```json(.*?)```", response, re.DOTALL)
            if not json_blocks:
                print("[ERROR] No JSON block found in response.")
                return None
            json_text = json_blocks[-1].strip()  # take the last JSON block
            data = json.loads(json_text)
            return data["CoT"]
        except Exception as e:
            print("[ERROR] Failed to parse CoT JSON:", e)
            return None

    def extract_final_conclusion(self, steps: List[Dict]) -> str:
        try:
            final = next(item for item in reversed(steps) if item["action"] == "Final Conclusion")
            return final["content"]
        except Exception as e:
            print("[ERROR] Failed to extract Final Conclusion:", e)
            return None

    def run(self):
        print("Loading dataset split...")
        dataset = load_dataset(self.dataset_name, name="explanation", split="train")
        #dataset = dataset.select(range(50)) # <-- Selects 50 instances from the dataset.


        for item in dataset:
            print(item["contest_number"])
            self.stats["total"] += 1
            if (self.explanation_type == "uncanny"):
                answer = item["image_uncanny_description"]
                #question = "Analyze the visual composition of the cartoon. What makes it feel surreal, exaggerated, or unusual?"
                question = "Analyze the visual scene and identify elements that contrast with what we expect in normal situations. Which visual or contextual contradictions might contribute to the cartoon’s humor or surreal effect?"
            elif (self.explanation_type == "canny"):
                answer = item["image_description"]
                question = "What do you observe in this image?"
            else:
                raise ValueError("Invalid explanation type. Choose either 'uncanny' or 'canny'.")
            print("MLLM device:", next(self.mllm.model.parameters()).device)
            print("Verifier device:", next(self.verifier.model.parameters()).device)
            print(f"\n[Q] {question}")
            prompt_text = self.format_prompt(question)

            response = self.mllm.prompt(
                image=item["image"],
                prompt=prompt_text,
                #fewshot=self.fewshot,
                temperature=0.7
            )

            steps = self.parse_cot_json(response)
            if not steps:
                print("[Skip] Could not parse CoT.")
                continue
            
            full_CoT = steps.copy()

            conclusion = self.extract_final_conclusion(steps)
            if not conclusion:
                print("[Skip] Could not extract final conclusion.")
                continue

            print("[Conclusion]", conclusion)
            is_correct = self.verifier.verify(conclusion, answer)
            print("[Verification]", is_correct)


            
            if not is_correct:
                #previous_conclusions = [conclusion]

                for strategy_name, strategy_template in random.sample(search_strategies, k=len(search_strategies)):
                    reasoning_so_far = json.dumps(steps[:-1], indent=2, ensure_ascii=False)

                    retry_prompt = strategy_template.format(question, reasoning_so_far)
                    
                    retry_response = self.mllm.prompt(
                        image=item["image"],
                        prompt=retry_prompt,
                        #fewshot=self.fewshot,
                        temperature=0.7
                    )
                    
                    retry_steps = self.parse_cot_json(retry_response)

                    if retry_steps:
                        print(f"\n[Retry Strategy: {strategy_name}] Prompt:\n{retry_prompt}\n")
                        
                        full_CoT.append({"action": "Verification", "content": f"Retrying with strategy: {strategy_name}"})
                        # Append new reasoning steps (skip first verification if duplicate)
                        full_CoT.extend(retry_steps[1:] if retry_steps[0]["action"] == "Verification" else retry_steps)

                        # Replace the last failed Verification step and extend CoT
                        steps = retry_steps
                        conclusion = self.extract_final_conclusion(steps)
                        
                        #if self.conclusion_is_reused(conclusion, previous_conclusions):
                            #print("[Skip] Repeated conclusion.")
                            #continue
                        #previous_conclusions.append(conclusion)


                        if conclusion:
                            is_correct = self.verifier.verify(conclusion, answer)
                            print("[New Conclusion]", conclusion)
                            print("[Verification after retry]", is_correct)
                            if is_correct:
                                break

            if is_correct:
                self.stats["correct"] += 1

            self.results.append({
                "instance_id": item["instance_id"],
                "contest_number": item["contest_number"],
                "question": question,
                "cot_response": json.dumps({"CoT": steps}, ensure_ascii=False, indent=2),
                "full_cot_response": json.dumps({"CoT": full_CoT}, ensure_ascii=False, indent=2),
                "conclusion": conclusion,
                "ground_truth": answer,
                "correct": is_correct,
                "retry_strategy": strategy_name if not is_correct else None
            })





    def format_prompt(self, question: str) -> str:
        return f"""<question>
    {question}
    </question>
    <|image|>

    Please first **analyze** the image in natural language using the Chain of Thought (CoT) method. 

    After completing your thinking, **output your result strictly** in the following JSON format:

    ```json
    {{
        "CoT": [
            {{"action": "Inner Thinking", "title": "...", "content": "..."}},
            ...,
            {{"action": "Final Conclusion", "content": "..."}},
            {{"action": "Verification", "content": "..."}}
        ]
    }}
    ```"""

    def save_results(self, path: str = "cot_results.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"[Saved] Results written to {path}")

