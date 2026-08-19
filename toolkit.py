"""
AI Productivity Toolkit
=======================
A modular Python framework implementing production prompt engineering,
zero-shot schema-enforced JSON extraction, code refactoring, and an
interactive Gradio interface using Hugging Face SmolLM2-1.7B-Instruct.

Author: Kunal Rawat
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr


class AIProductivityToolkit:
    """Core engine handling prompt templates, inference calibration, and structured text generation."""

    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[*] Initializing model on device: {self.device.upper()} ({self.dtype})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def _execute_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Formats and executes prompts using model-specific ChatML templates."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = self.pipe(formatted_prompt)[0]["generated_text"]
        
        if "<|im_start|>assistant" in response:
            clean_output = response.split("<|im_start|>assistant\n")[-1]
            return clean_output.replace("<|im_end|>", "").strip()
        return response.split(user_prompt)[-1].strip()

    def summarize_executive(self, text: str) -> str:
        """Condenses complex technical text into 3 bullet points and a key takeaway."""
        system_prompt = "You are an expert technical editor. Summarize dense information with extreme precision."
        user_prompt = (
            f"Summarize the following text into exactly 3 concise, high-signal bullet points followed by a "
            f"single-sentence 'Key Takeaway:'. Do not copy the text verbatim.\n\nText:\n{text}"
        )
        return self._execute_chat(system_prompt, user_prompt)

    def extract_structured_entities(self, text: str) -> str:
        """Extracts entities from raw text into a validated JSON schema."""
        system_prompt = "You are a data extraction engine. Extract information into strictly valid JSON."
        user_prompt = (
            f"Extract entities from the provided text into this exact JSON schema:\n"
            f'{{"candidate_name": "string", "primary_skills": ["skill1", "skill2"], "experience_years": "string", "target_role": "string"}}\n\n'
            f"Text:\n{text}\n\nOutput only raw JSON, no conversational filler or markdown formatting."
        )
        return self._execute_chat(system_prompt, user_prompt)

    def code_debugger_refactor(self, code: str, issue_description: str = "Identify bugs and optimize logic") -> str:
        """Analyzes syntax/logic flaws and provides a refactored solution."""
        system_prompt = "You are a senior software engineer specializing in algorithmic refactoring."
        user_prompt = (
            f"Analyze this Python snippet, identify the logic flaw, provide the fixed version, and explain the fix in 2 sentences.\n\n"
            f"Code:\n```python\n{code}\n```\nIssue/Context: {issue_description}"
        )
        return self._execute_chat(system_prompt, user_prompt)

    def cold_outreach_generator(
        self, recipient_name: str, company: str, objective: str, background: str
    ) -> str:
        """Generates concise, value-oriented networking emails."""
        system_prompt = "You are a professional communication strategist. Write concise, high-conversion networking emails."
        user_prompt = (
            f"Write a value-focused cold networking email under 100 words.\n"
            f"Recipient: {recipient_name}\n"
            f"Company: {company}\n"
            f"Objective: {objective}\n"
            f"Candidate Background: {background}"
        )
        return self._execute_chat(system_prompt, user_prompt)


def launch_web_ui(toolkit: AIProductivityToolkit, share: bool = True):
    """Builds and launches an interactive Gradio web application."""
    
    def handle_request(task, text_input, recipient, company, objective, background):
        if not text_input and task != "Cold Outreach Generator":
            return "Error: Input text/code cannot be empty."
            
        if task == "Executive Summarizer":
            return toolkit.summarize_executive(text_input)
        elif task == "JSON Entity Extractor":
            return toolkit.extract_structured_entities(text_input)
        elif task == "Code Debugger & Refactor":
            return toolkit.code_debugger_refactor(text_input)
        elif task == "Cold Outreach Generator":
            return toolkit.cold_outreach_generator(recipient, company, objective, background)
        return "Unknown task selected."

    with gr.Blocks(title="AI Productivity Toolkit") as demo:
        gr.Markdown("# 🚀 AI Productivity Toolkit")
        gr.Markdown(
            "An open-source Small Language Model (SLM) pipeline for automated developer workflows, "
            "structured JSON generation, and code refactoring."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                task_dropdown = gr.Dropdown(
                    choices=[
                        "Executive Summarizer",
                        "JSON Entity Extractor",
                        "Code Debugger & Refactor",
                        "Cold Outreach Generator"
                    ],
                    value="Executive Summarizer",
                    label="Select Workflow Tool"
                )
                
                input_text = gr.Textbox(
                    lines=8,
                    label="Input Payload (Text / Code / Raw Context)",
                    placeholder="Paste your source text or code here..."
                )
                
                with gr.Accordion("Outreach Email Parameters (Outreach Generator only)", open=False):
                    recipient = gr.Textbox(label="Recipient Name / Title", placeholder="e.g., Lead AI Researcher")
                    company = gr.Textbox(label="Target Organization", placeholder="e.g., Applied AI Labs")
                    objective = gr.Textbox(label="Core Objective", placeholder="e.g., Research Internship")
                    background = gr.Textbox(label="Your Background", placeholder="e.g., B.Tech AIML Student")
                
                submit_btn = gr.Button("Execute Workflow", variant="primary")
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(lines=12, label="Generated Model Output", interactive=False)

        submit_btn.click(
            fn=handle_request,
            inputs=[task_dropdown, input_text, recipient, company, objective, background],
            outputs=output_text
        )

    demo.launch(share=share)


if __name__ == "__main__":
    app_toolkit = AIProductivityToolkit()
    launch_web_ui(app_toolkit, share=True)
