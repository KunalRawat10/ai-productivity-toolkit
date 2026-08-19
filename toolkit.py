"""
AI Productivity Toolkit
Modular prompt engineering workflows for structured extraction, summarization, and refactoring.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class AIProductivityToolkit:
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=250, temperature=0.2)

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self.pipe(formatted)[0]["generated_text"]
        return out.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()

    def summarize(self, text: str) -> str:
        return self._generate("Summarize the text into 3 technical bullet points and a one-sentence takeaway.", text)

    def extract_json(self, text: str) -> str:
        return self._generate("Extract data into strictly valid JSON schema (candidate_name, skills, experience).", text)
