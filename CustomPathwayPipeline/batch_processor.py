import queue
import threading
import time
import asyncio
from langchain_core.messages import AIMessage
from config import MAX_BATCH_SIZE, BATCH_TIMEOUT_SECONDS
from loader import llm_model_loader # Import the instantiated model loader

# Global queue and futures for batching
request_queue = queue.Queue()
response_futures = {}  # id -> asyncio.Future

class BatchProcessor:
    """
    Manages the batching of LLM inference requests.
    """
    def __init__(self):
        self.llm_model_loader = llm_model_loader
        # Ensure a default event loop is set for `asyncio.create_future` to work
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        # Start the batch processing thread when the processor is initialized
        self.batch_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.batch_thread.start()
        print("Batch processing thread started.")

    def _processing_loop(self):
        while True:
            batch = []
            # Get the first item with a timeout
            try:
                item = request_queue.get(timeout=BATCH_TIMEOUT_SECONDS)
                batch.append(item)
            except queue.Empty:
                # If queue is empty, wait for a bit and then check again
                time.sleep(0.01) # Small sleep to prevent busy-waiting
                continue # Continue to the next iteration to check for new items
            
            start_time = time.time()
            # Collect more items until MAX_BATCH_SIZE or BATCH_TIMEOUT_SECONDS
            while len(batch) < MAX_BATCH_SIZE and (time.time() - start_time) < BATCH_TIMEOUT_SECONDS:
                try:
                    item = request_queue.get(timeout=0.001) # Small timeout to avoid blocking
                    batch.append(item)
                except queue.Empty:
                    break # No more items, process current batch
                
            if not batch:
                continue # Nothing to process, go back to waiting
            
            prompts_to_process = [item['formatted_prompt'] for item in batch]
            request_ids = [item['request_id'] for item in batch]
            user_memories = [item['user_memory'] for item in batch]
            user_messages = [item['user_message'] for item in batch]
            
            try:
                batch_responses = self.llm_model_loader.generate_batched_responses(prompts_to_process)
                
                # Distribute responses back to their respective futures
                for i, response in enumerate(batch_responses):
                    req_id = request_ids[i]
                    # original_user_message = user_messages[i] # Not directly used here, but good to have if needed
                    original_user_memory = user_memories[i]

                    # Add to Langchain memory
                    original_user_memory.chat_memory.add_message(AIMessage(content=response))
                    
                    # Set the result on the future
                    if req_id in response_futures:
                        response_futures[req_id].set_result(response)
                        print(f" [Batch] Set result for request_id: {req_id}")
                    else:
                        print(f" [Batch] Warning: Future not found for request_id: {req_id}")
                        
            except Exception as e:
                print(f" [Batch] Error during batch generation: {e}")
                # If an error occurs, set exception on all futures in the batch
                for i, req_id in enumerate(request_ids):
                    if req_id in response_futures:
                        response_futures[req_id].set_exception(e)

    def add_request_to_queue(self, request_data: dict, future: asyncio.Future):
        """
        Adds a new request to the batch processing queue.
        """
        request_queue.put(request_data)
        response_futures[request_data['request_id']] = future

    def get_future_result(self, request_id: str) -> tuple:
        """
        Checks the status of a request's future.
        Returns (status, result/error_message).
        """
        future = response_futures.get(request_id)
        if not future:
            return "not_found", "Request ID not found or already processed"
        
        if future.done():
            try:
                result = future.result()
                del response_futures[request_id] # Clean up
                return "completed", result
            except Exception as e:
                del response_futures[request_id] # Clean up
                return "error", str(e)
        else:
            return "processing", None

# Instantiate the batch processor
batch_processor = BatchProcessor()