from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

def get_embedder(model="intfloat/e5-large-v2", device="cuda:0"):
    return SentenceTransformerEmbedder(model=model, device=device)
