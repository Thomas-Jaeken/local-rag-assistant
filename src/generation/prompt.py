from typing import List, Dict

def build_prompt(query: str, retrieved_chunks: List[Dict], max_context_chars=2500) -> str:
    """
    Create a prompt for the LLM that includes the top-k retrieved chunks.
    """
    context_texts = []
    char_count = 0
    print("retrieved "+ str(len(retrieved_chunks))+ " chuncks")
    for chunk in retrieved_chunks:
        text = chunk["text"]
        print(len(text),max_context_chars)
        if char_count + len(text) > max_context_chars:
            break
        context_texts.append(text)
        char_count += len(text)

    context_str = "\n\n".join(context_texts)

    # prevent silent failure of context retrieval
    if not context_str.strip():
        raise RuntimeError("Context is empty — retrieval failed")
    
    prompt = f"Instruct: You are a helpful research assistent who answers a scientist while trying to use the given context. If you cannot find the answer in the context reply 'Sorry don't have that detail'. Given the context '{context_str}', answer this:{query}\nOutput:"
    # prompt = (f"I am expert AI assistant and will tell you what I know about {query}. ")
    
    return prompt
