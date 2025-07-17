from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class LLMModelLoader:
    """
        Initializes the LLMModelLoader.

        Args:
            model_name (str): The name or path of the pre-trained model to load.
            device (str): The device to load the model onto (e.g., "cuda", "cpu").
            generation_args (dict): Dictionary of arguments for model.generate().
            quantization_config (BitsAndBytesConfig): Optional quantization configuration.
    """
    def __init__(self, model_name: str, device: str, generation_args: dict, quantization_config: BitsAndBytesConfig):
        self.model_name = model_name
        self.current_device = device

        self.tokenizer = self._load_tokenizer()

        # Use provided quantization_config or default to BitsAndBytesConfig
        self.bnb_config = quantization_config if quantization_config else BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = self._load_model()

        # Use provided generation_args or default, then add pad_token_id
        self.generation_args = generation_args.copy()
        self.generation_args["pad_token_id"] = self.tokenizer.eos_token_id


    def _load_tokenizer(self):
        """
        Loads the tokenizer from the specified model name.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer for {self.model_name}: {e}")
            raise

    def _load_model(self):
        """
        Loads the language model from the specified model name to the current device,
        with optional quantization. Falls back to CPU if CUDA is unavailable or fails.
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.bnb_config,
                trust_remote_code=True
            ).to(self.current_device)
            print(f"Model loaded to {self.current_device}.")
        except Exception as e:
            print(f"CUDA not available or error loading to {self.current_device}: {e}. Falling back to CPU.")
            model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to("cpu")
            self.current_device = "cpu" # Update device to reflect actual loaded device

        model.config.pad_token_id = model.config.eos_token_id
        model.eval() # Set model to evaluation mode
        return model

    def generate_batched_responses(self, prompts: list[str]) -> list[str]:
        """
        Generates responses for a batch of prompts using the loaded LLM.

        Args:
            prompts (list[str]): A list of input prompt strings.

        Returns:
            list[str]: A list of generated response strings.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not loaded. Call _load_model() and _load_tokenizer() first.")

        print(f" [Batch] Processing batch of {len(prompts)} prompts...")

        inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.current_device)

        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = self.model.generate(**inputs, **self.generation_args)

        decoded_responses = []
        for output in outputs:
            # Decode the full output and extract only the generated response
            decoded = self.tokenizer.decode(output, skip_special_tokens=False)
            # Assuming a conversation template like "[INST] prompt [/INST] response"
            # Adjust splitting logic if your prompt template is different
            response_text = decoded.split("[/INST]")[-1].strip()
            decoded_responses.append(response_text)

        print(f" [Batch] Finished processing batch.")
        return decoded_responses

# Example of how to use it with default settings (similar to original usage)
# llm_model_loader = LLMModelLoader()

# Example of how to use it with custom settings
# from transformers import BitsAndBytesConfig
# custom_bnb_config = BitsAndBytesConfig(load_in_8bit=True)
# custom_generation_args = {"max_new_tokens": 100, "temperature": 0.5}
# custom_loader = LLMModelLoader(
#     model_name="google/flan-t5-small", # Example: A different model
#     device="cpu", # Example: Force CPU
#     generation_args=custom_generation_args,
#     quantization_config=custom_bnb_config
# )