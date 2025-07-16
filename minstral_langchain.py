# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.vectorstores import PathwayVectorClient
from langchain.memory import ConversationBufferMemory
import torch, uvicorn, os, datetime

MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
PORT       = int(os.getenv("PORT", 8011))

# --- Quantised model loading ---
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                llm_int8_threshold=6.0,
                                bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True).eval()

DEVICE = next(model.parameters()).device

# --- Retrieval ---
vector_client = PathwayVectorClient(host="127.0.0.1", port=8666)
retriever     = vector_client.as_retriever()

memory = ConversationBufferMemory(return_messages=True)
app    = FastAPI(title="Mistral RAG Chatbot")

class ChatPayload(BaseModel):
    messages: str

def build_prompt(question: str, context: str, history: str) -> str:
    # Clean the context by removing leading dots and extra whitespace
    cleaned_context = "\n".join(line.lstrip(". ").strip() for line in context.split("\n") if line.strip())
    
    user_block = (f"Answer the question using ONLY the following context. If the answer isn't in the context, say 'I don't know'.\n"
                 f"Question: {question}\n\n"
                 f"Context:\n{cleaned_context}\n\n"
                 f"Answer directly and concisely with just the requested information. \n\n **Important Note** :" 
                f"- Only use the context and the chat history to answer. If you don't know the answer, say so politely.."
                f"- Only specify the time/date if the venue/date specified in the context is near to the current date, else ask them to check the website.")
    print(f" user_block = {user_block}")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_block}],
        tokenize=False,
        add_generation_prompt=True)

def generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs,
                             max_new_tokens=512,
                             temperature=0.7, top_p=0.9,
                             repetition_penalty=1.1,
                             use_cache=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/")
def chat(payload: ChatPayload):
    if not payload.messages:
        raise HTTPException(400, "Empty message")

    memory.chat_memory.add_ai_message(payload.messages)
    ctx_docs = retriever.invoke(payload.messages)[:3]
    context = "\n\n".join(d.page_content for d in ctx_docs)
    # print(f"context is {context}")
    history = "\n".join(f"{m.type}: {m.content}" for m in memory.chat_memory.messages[-6:])

    prompt = build_prompt(payload.messages, context, history)
    answer_raw = generate(prompt)
    
    # Clean up the response
    answer = answer_raw.split("[/INST]")[-1].strip()
    answer = answer.split("\n")[0].strip()  # Take only the first line
    
    # Get source filenames, handling None cases
    sources = []
    for d in ctx_docs:
        source = d.metadata.get("file_name") if hasattr(d, 'metadata') else None
        if source:
            sources.append(source)
    
    memory.chat_memory.add_user_message(answer)
    return {"response": answer,
            "sources": sources if sources else ["No source available"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
