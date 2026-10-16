import os

os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache/models"
os.environ["HF_HUB_CACHE"] = "D:/huggingface_cache/hub"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

prompt = """<|system|>
You are a helpful assistant.</s>
<|user|>
What is the capital of Pakistan?</s>
<|assistant|>
"""

result = pipe(prompt)

print(result[0]["generated_text"])