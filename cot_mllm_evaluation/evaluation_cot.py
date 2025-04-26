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
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps in detail and provide the final answer (not generic). Please give details of the previous 'Inner Thinking' steps. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning using **backtracking** to revisit earlier points of reasoning and construct a new Final Conclusion.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
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
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps in detail and provide the final answer (not generic). Please give details of the previous 'Inner Thinking' steps. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by exploring new approaches to solving this problem and construct a new Final Conclusion.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
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
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps in detail and provide the final answer (not generic). Please give details of the previous 'Inner Thinking' steps. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by conducting a thorough **validation** process to ensure validity and construct a new Final Conclusion.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""),

    ("Correction", """<question>
{}
</question>
<|image|>
<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps in detail and provide the final answer (not generic). Please give details of the previous 'Inner Thinking' steps. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by making precise **corrections** to address prior flaws and construct a new Final Conclusion.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```""")
]


class CoTEvaluator:
    def __init__(self, dataset_name: str, mllm: HuggingFaceMLLM, verifier: LLMVerifier, fewshot: List[Dict] = None):
        self.dataset_name = dataset_name
        self.mllm = mllm
        self.verifier = verifier
        self.fewshot = fewshot or []
        self.stats = {"correct": 0, "total": 0}
        self.results = []
    
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

        for item in dataset:
            self.stats["total"] += 1
            answer = item["image_uncanny_description"]
            #question =  "You are an art critic. Write an uncanny literal description of the cartoon.\n"
            question = "What is uncanny about the image?"
            print(f"\n[Q] {question}")
            prompt_text = self.format_prompt(question)

            response = self.mllm.prompt(
                image=item["image"],
                prompt=prompt_text,
                fewshot=self.fewshot
            )

            print(response)
            steps = self.parse_cot_json(response)
            if not steps:
                print("[Skip] Could not parse CoT.")
                continue

            conclusion = self.extract_final_conclusion(steps)
            if not conclusion:
                print("[Skip] Could not extract final conclusion.")
                continue

            print("[Conclusion]", conclusion)
            is_correct = self.verifier.verify(conclusion, answer)
            print("[Verification]", is_correct)

            if not is_correct:
                for strategy_name, strategy_template in random.sample(search_strategies, k=len(search_strategies)):
                    reasoning_so_far = json.dumps(steps[:-1], indent=2)
                    retry_prompt = strategy_template.format(question, reasoning_so_far)
                    
                    retry_response = self.mllm.prompt(
                        image=item["image"],
                        prompt=retry_prompt,
                        fewshot=self.fewshot
                    )
                    
                    retry_steps = self.parse_cot_json(retry_response)

                    if retry_steps:
                        print(f"[Retry with {strategy_name}]")
                        # Replace the last failed Verification step and extend CoT
                        steps = steps[:-1] + retry_steps
                        conclusion = self.extract_final_conclusion(steps)

                        if conclusion:
                            is_correct = self.verifier.verify(conclusion, answer)
                            print("[New Conclusion]", conclusion)
                            print("[Verification after retry]", is_correct)
                            if is_correct:
                                break

            if is_correct:
                self.stats["correct"] += 1

            self.results.append({
                "question": question,
                "cot_response": json.dumps({"CoT": steps}, ensure_ascii=False, indent=2),
                "conclusion": conclusion,
                "ground_truth": answer,
                "correct": is_correct
            })




    def format_prompt(self, question: str) -> str:
        return f"""<question>
    {question}
    </question>
    <|image|>

    Please first **analyze** the image in natural language using the Chain of Thought (CoT) method. 
    (Your 'Final Conclusion' should summarize and combine the 'Inner Thinking' steps.)

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

