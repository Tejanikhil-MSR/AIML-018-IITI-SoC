import queue
import threading
import time
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from config import MAX_BATCH_SIZE, BATCH_TIMEOUT_SECONDS
from loader import llm_model_loader

request_queue = queue.Queue()
response_futures = {}  # id -> asyncio.Future

class BatchProcessor:
    """
    Manages the batching of LLM inference requests and updates user memories.
    """
    def __init__(self):
        self.llm_model_loader = llm_model_loader
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        self.batch_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.batch_thread.start()
        print("Batch processing thread started.")

    def _processing_loop(self):
        while True:
            batch = []
            try:
                item = request_queue.get(timeout=BATCH_TIMEOUT_SECONDS)
                batch.append(item)
            except queue.Empty:
                time.sleep(0.01)
                continue
            
            start_time = time.time()
            while len(batch) < MAX_BATCH_SIZE and (time.time() - start_time) < BATCH_TIMEOUT_SECONDS:
                try:
                    item = request_queue.get(timeout=0.001)
                    batch.append(item)
                except queue.Empty:
                    break
                
            if not batch:
                continue
            
            prompts_to_process = []
            request_data_for_future = []

            for item in batch:
                prompts_to_process.append(item['formatted_prompt'])
                # Reconstruct ConversationBufferMemory for each item in the batch
                current_memory = ConversationBufferMemory(return_messages=True)
                for msg_data in item['initial_chat_messages']:
                    if msg_data['type'] == 'human':
                        current_memory.chat_memory.add_message(HumanMessage(content=msg_data['content']))
                    elif msg_data['type'] == 'ai':
                        current_memory.chat_memory.add_message(AIMessage(content=msg_data['content']))
                
                # We will add the HumanMessage here so it's part of the memory before AI responds
                current_memory.chat_memory.add_message(HumanMessage(content=item['user_message']))

                request_data_for_future.append({
                    'request_id': item['request_id'],
                    'user_memory_instance': current_memory, 
                    'initial_user_message': item['user_message']
                })
            
            # --- Perform actual LLM generation ---
            try:
                batch_responses = self.llm_model_loader.generate_batched_responses(prompts_to_process)
                
                for i, response_text in enumerate(batch_responses):
                    req_data = request_data_for_future[i]
                    req_id = req_data['request_id']
                    user_memory_instance = req_data['user_memory_instance']

                    user_memory_instance.chat_memory.add_message(AIMessage(content=response_text))
                    print(f" [Batch] Updated memory for request_id: {req_id}")

                    serializable_messages = []
                    for msg in user_memory_instance.chat_memory.messages:
                        if isinstance(msg, HumanMessage):
                            serializable_messages.append({'type': 'human', 'content': msg.content})
                        elif isinstance(msg, AIMessage):
                            serializable_messages.append({'type': 'ai', 'content': msg.content})

                    if req_id in response_futures:
                        response_futures[req_id].set_result({
                            'response': response_text,
                            'updated_chat_messages': serializable_messages
                        })
                        print(f" [Batch] Set result for request_id: {req_id} with updated messages.")
                    else:
                        print(f" [Batch] Warning: Future not found for request_id: {req_id}")
                        
            except Exception as e:
                print(f" [Batch] Error during batch generation: {e}")
                for i, req_data in enumerate(request_data_for_future):
                    req_id = req_data['request_id']
                    if req_id in response_futures:
                        response_futures[req_id].set_exception(str(e))


    def add_request_to_queue(self, request_data: dict, future: asyncio.Future):
        """
        Adds a new request to the batch processing queue.
        `request_data` should include 'formatted_prompt', 'initial_chat_messages', 'user_message'.
        """
        request_queue.put(request_data)
        response_futures[request_data['request_id']] = future

    def get_future_result(self, request_id: str) -> tuple:
        """
        Checks the status of a request's future.
        Returns (status, result_data/error_message).
        """
        future = response_futures.get(request_id)
        if not future:
            return "not_found", "Request ID not found or already processed"
        
        if future.done():
            try:
                result_data = future.result() 
                del response_futures[request_id]
                return "completed", result_data
            except Exception as e:
                del response_futures[request_id]
                return "error", str(e)
        else:
            return "processing", None


batch_processor = BatchProcessor()
