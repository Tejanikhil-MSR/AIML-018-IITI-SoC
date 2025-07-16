from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from config import MODEL_CHOICE, MODEL_REGISTRY, DEVICE, GENERATION_ARGS

class LLMModelLoader:
    """
    Handles loading and configuration of the Language Model and its tokenizer.
    """
    def __init__(self):
        self.model_name = MODEL_REGISTRY[MODEL_CHOICE]
        self.current_device = DEVICE 
        self.tokenizer = self._load_tokenizer()
        self.bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                llm_int8_threshold=6.0,
                                bnb_4bit_compute_dtype=torch.float16)
        self.model = self._load_model()
        self.generation_args = GENERATION_ARGS.copy()
        self.generation_args["pad_token_id"] = self.tokenizer.eos_token_id
        

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 
        tokenizer.truncation_side = "left"
        return tokenizer

    def _load_model(self):
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=self.bnb_config, trust_remote_code=True).to(self.current_device)
            print(f"Model loaded to {self.current_device}.")
        except Exception as e:
            print(f"CUDA not available or error loading to GPU: {e}. Falling back to CPU.")
            model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to("cpu")
            self.current_device = "cpu"
            
        model.config.pad_token_id = model.config.eos_token_id
        model.eval()
        return model

    def generate_batched_responses(self, prompts: list[str]) -> list[str]:
        """
        Generates responses for a batch of prompts using the loaded LLM.
        """
        print(f" [Batch] Processing batch of {len(prompts)} prompts...")

        inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.current_device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generation_args)

        decoded_responses = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=False)
            print(f"decoded res: {decoded}")
            response_text = decoded.split("[/INST]")[-1].strip()
            decoded_responses.append(response_text)
        
        print(f" [Batch] Finished processing batch.")
        return decoded_responses

llm_model_loader = LLMModelLoader()