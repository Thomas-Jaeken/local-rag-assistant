import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="microsoft/phi-2",
    dtype=torch.bfloat16,
    device_map="auto",
)

# result = pipe("Explain quantum computing simply. ")
query="which wavelength range allows atmospheric transmission"
result = pipe(f"I am expert AI assistant and will tell you what I know about {query}. "
    )
print(result[0]["generated_text"][])