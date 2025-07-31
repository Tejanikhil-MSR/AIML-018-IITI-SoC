import queue
import threading
import time
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from typing import Any

request_queue = queue.Queue()
response_futures = {}  # id -> asyncio.Future

class BatchProcessor:
    def __init__(self, llm_generator: Any, max_batch_size: int, batch_timeout_seconds: float):
        """
        Initializes and starts the BatchProcessor with 1 thread.

        Args:
            llm_generator (any): An object with a 'generate_batched_responses' method
                                 (e.g., LLMModelLoader instance). Defaults to llm_model_loader.
            max_batch_size (int): The maximum number of prompts to process in a single batch.
                                  Defaults to MAX_BATCH_SIZE from config.
            batch_timeout_seconds (float): The maximum time to wait for a batch to fill
                                           before processing. Defaults to BATCH_TIMEOUT_SECONDS from config.
        """
        self.llm_generator = llm_generator
        self.max_batch_size = max_batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        self.batch_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.batch_thread.start()

        print("Batch processing thread started.")

    def _processing_loop(self):
        """
        The function that takes requests from queues and process them in batches.
        """
        while True:
            batch = []
            # === Step 1.0 : Load the first request from the batch ===
            try:
                # Get the first item with a timeout to prevent infinite blocking
                item = request_queue.get(timeout=self.batch_timeout_seconds)
                batch.append(item)
            except queue.Empty:
                # If queue is empty, wait a short moment and continue
                time.sleep(0.01)
                continue

            start_time = time.time()
            # === Step 1.1 : Fill the batch with the subsequent items from the request queue until max size or timeout ===
            while len(batch) < self.max_batch_size and (time.time() - start_time) < self.batch_timeout_seconds:
                try:
                    # Get subsequent items without blocking for too long
                    item = request_queue.get(timeout=0.001)
                    batch.append(item)
                except queue.Empty:
                    break # No more items in queue, process current batch
            
            print(f"Processing a batch of size {len(batch)}...")

            prompts_to_process = []
            request_data_for_future = []

            # === Step 2 : Prepare prompts and user memory instances for each item in the batch ===
            for item in batch:
                prompts_to_process.append(item['formatted_prompt'])
                # For loading the previous messages from the session history
                current_memory = ConversationBufferMemory(return_messages=True)
                for msg_data in item['initial_chat_messages']:
                    if msg_data['type'] == 'human':
                        current_memory.chat_memory.add_message(HumanMessage(content=msg_data['content']))
                    # elif msg_data['type'] == 'ai':
                        # current_memory.chat_memory.add_message(AIMessage(content=msg_data['content']))

                # Instead of adding AI generated response to the chat_memory add the keywords from the retrieved documents
                # concatenated_keywords = "\n".join(
                #     [f"Doc-{i+1} Retrieved keywords : {', '.join(keywords)}" for i, keywords in enumerate(item['keywords'])]
                # )

                # current_memory.chat_memory.add_message(AIMessage(content="Previous Response Keywords : " + concatenated_keywords)) 

                request_data_for_future.append({
                    'request_id': item['request_id'],
                    'user_memory_instance': current_memory,
                    'initial_user_message': item['user_message'],
                })

            # === Step 3 : Generate the responses by processing the requests in branch ===
            try:
                batch_responses = self.llm_generator.generate_batched_responses(prompts_to_process)
                for i, response_text in enumerate(batch_responses):
                    req_data = request_data_for_future[i]
                    req_id = req_data['request_id']
                    user_memory_instance = req_data['user_memory_instance']

                    # === Step 4 : Serialize the messages for storage (in Flask session) ===
                    serializable_messages = []
                    for msg in user_memory_instance.chat_memory.messages:
                        if isinstance(msg, HumanMessage):
                            serializable_messages.append({'type': 'human', 'content': msg.content})
                        # elif isinstance(msg, AIMessage):
                        #     serializable_messages.append({'type': 'ai', 'content': msg.content})

                    # === Step 5 : Update the response future of the processed request id with AI generated response and updated user memory ===
                    if req_id in response_futures:
                        response_futures[req_id].set_result({
                            'response': response_text,
                            'updated_chat_messages': serializable_messages,
                        })
                        print(f" [Batch] Set result for request_id: {req_id} with updated messages.")
                    else:
                        print(f" [Batch] Warning: Future not found for request_id: {req_id}")

            except Exception as e:
                print(f" [Batch] Error during batch generation: {e}")
                for i, req_data in enumerate(request_data_for_future):
                    req_id = req_data['request_id']
                    if req_id in response_futures:
                        response_futures[req_id].set_exception(e)

    def add_request_to_queue(self, request_data: dict, future: asyncio.Future):
        """
        Adds a new request to the batch processing queue.
        `request_data` should include 'formatted_prompt', 'initial_chat_messages', 'user_message'.
        It can also include other metadata relevant to the request.
        """
        request_queue.put(request_data)
        response_futures[request_data['request_id']] = future

    def get_future_result(self, request_id: str) -> tuple: # called by the flask API
        """
            Checks the status of a request's future.
            Returns (status, result_data/error_message).
        """
        future = response_futures.get(request_id)
        print(response_futures,request_id)

        if not future:
            return "not_found", "Request ID not found or already processed"

        if future.done():
            try:
                result_data = future.result()
                del response_futures[request_id] # Clean up future once result is retrieved
                return "completed", result_data
            except Exception as e:
                del response_futures[request_id] # Clean up future even on error
                return "error", str(e)
        else:
            return "processing", None

# Example of how to use it with custom settings
# class MyCustomLLMGenerator:
#     def generate_batched_responses(self, prompts: list[str]) -> list[str]:
#         print("Custom LLM generating responses...")
#         return [f"Custom response for: {p}" for p in prompts]
#
# custom_llm_gen = MyCustomLLMGenerator()
# custom_batch_processor = BatchProcessor(
#     llm_generator=custom_llm_gen,
#     max_batch_size=2,
#     batch_timeout_seconds=0.1
# )