from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
def generate_answer(prompt: str, model_name="microsoft/phi-2", max_tokens=200) -> str:
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # # model = AutoModelForCausalLM.from_pretrained(
    # #     model_name,
    # #     device_map="auto",
    # #     dtype="auto"
    # # )

    # generator = pipeline(
    #     task="text-generation",
    #     model="microsoft/phi-2",
    #     device=0,
    #     # tokenizer=tokenizer
    # )
    # # sampling was unstable on mac with fp16 so i chose to decode determinstically. This means always the most likely next character is picked without inject some randomness and sometimes choosing a slightly less likely character. 
    # # this risks getting stuck in a loop 

    # # output = generator(
    # #     prompt,
    # #     max_new_tokens=max_tokens,
    # #     do_sample=False,
    # #     repetition_penalty=1.5
    # # )
    # prompt = "Instruct: Write a detailed analogy between mathematics and a lighthouse. Output:"
    
    generator = pipeline(
    task="text-generation",
    model=model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    )
    output = generator(prompt)
    return output[0]["generated_text"][len(prompt):]
