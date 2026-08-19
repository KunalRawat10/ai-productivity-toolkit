"""
AI Productivity Toolkit
=======================
A modular Python framework implementing production-grade prompt engineering,
structured schema extraction, and automated developer workflows using
open-weights Small Language Models (SLMs) and Gradio.

Author: Kunal Rawat
"""

import os
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr


class AIProductivityToolkit:
    """Core engine handling prompt templates, inference calibration, and structured text generation."""

    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
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
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def _execute_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Formats and executes a conversational prompt using model-specific ChatML templates."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = self.pipe(formatted_prompt)[0]["generated_text"]
        
        # Clean ChatML artifacts from response
        if "<|im_start|>assistant" in response:
            clean_output = response.split("<|im_start|>assistant\n")[-1]
            return clean_output.replace("<|im_end|>", "").strip()
        return response.split(user_prompt)[-1].strip()

    def summarize_executive(self, text: str) -> str:
        """Condenses complex technical literature into bullet points and a core takeaway."""
        system_prompt = (
            "You are an expert technical editor. Summarize the provided text into exactly 3 crisp, "
            "informative bullet points followed by a single-sentence 'Key Takeaway:'. Do not add conversational filler."
        )
        return self._execute_chat(system_prompt, text)

    def extract_structured_entities(self, text: str) -> str:
        """Extracts named entities from raw text into a strict, validated JSON schema."""
        system_prompt = (
            "You are an automated data extraction engine. Extract structured entities from the input text "
            "and output strictly valid JSON matching this schema:\n"
            "{\n"
            '  "candidate_name": "string",\n'
            '  "primary_skills": ["skill1", "skill2"],\n'
            '  "experience_years": "string or int",\n'
            '  "target_role": "string"\n'
            "}\n"
            "Output raw JSON only with no conversational text or markdown code blocks."
        )
        return self._execute_chat(system_prompt, text)

    def code_debugger_refactor(self, code: str, issue_description: str = "Identify bugs and optimize logic") -> str:
        """Analyzes syntax/logic flaws in code snippets and returns a refactored implementation."""
        system_prompt = (
            "You are a senior software engineer. Analyze the code, identify bugs or suboptimal logic, "
            "provide the corrected code snippet, and explain the fix in two concise sentences."
        )
        user_prompt = f"Code:\n```python\n{code}\n```\nIssue/Context: {issue_description}"
        return self._execute_chat(system_prompt, user_prompt)

    def cold_outreach_generator(
        self, recipient_name: str, company: str, objective: str, background: str
    ) -> str:
        """Generates concise, value-oriented professional cold outreach emails."""
        system_prompt = (
            "You are a professional communication strategist. Write a high-conversion cold networking email. "
            "Keep the response under 100 words. Maintain a confident, polite, and value-first tone."
        )
        user_prompt = (
            f"Recipient: {recipient_name}\n"
            f"Target Organization: {company}\n"
            f"Objective: {objective}\n"
            f"Candidate Background: {background}"
        )
        return self._execute_chat(system_prompt, user_prompt)


def launch_web_ui(toolkit: AIProductivityToolkit, share: bool = True):
    """Builds and launches a full-stack Gradio web interface."""
    
    def handle_request(task, text_input, recipient, company, objective, background):
        if not text_input and task != "Cold Outreach Generator":
            return "Error: Input text/code cannot be empty."
            
        if task == "Executive Technical Summarizer":
            return toolkit.summarize_executive(text_input)
        elif task == "Structured JSON Extractor":
            return toolkit.extract_structured_entities(text_input)
        elif task == "Code Debugger & Refactorer":
            return toolkit.code_debugger_refactor(text_input)
        elif task == "Cold Outreach Generator":
            return toolkit.cold_outreach_generator(recipient, company, objective, background)
        return "Unknown task selected."

    with gr.Blocks(title="AI Productivity Toolkit") as demo:
        gr.Markdown("# 🚀 AI Productivity Toolkit")
        gr.Markdown(
            "An open-source, small language model (SLM) pipeline for automated developer workflows, "
            "structured generation, and zero-shot entity extraction."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                task_dropdown = gr.Dropdown(
                    choices=[
                        "Executive Technical Summarizer",
                        "Structured JSON Extractor",
                        "Code Debugger & Refactorer",
                        "Cold Outreach Generator"
                    ],
                    value="Executive Technical Summarizer",
                    label="Select Workflow Tool"
                )
                
                input_text = gr.Textbox(
                    lines=8,
                    label="Input Payload (Text / Code / Raw Context)",
                    placeholder="Paste your source text or code here..."
                )
                
                with gr.Accordion("Outreach Email Parameters (Outreach Generator only)", open=False):
                    recipient = gr.Textbox(label="Recipient Name / Title", placeholder="e.g., Hiring Manager")
                    company = gr.Textbox(label="Target Organization", placeholder="e.g., AI Labs")
                    objective = gr.Textbox(label="Core Objective", placeholder="e.g., Summer Research Internship")
                    background = gr.Textbox(label="Your Background", placeholder="e.g., Undergraduate AIML Student")
                
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
