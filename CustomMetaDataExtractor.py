def custom_metadata_extractor(chunk, current_metadata):
    file_name = current_metadata.get("file_name", "unknown_file")
    
    # Initialize or increment chunk counter for this file
    if file_name not in chunk_counter_per_file:
        chunk_counter_per_file[file_name] = 0
        
    chunk_counter_per_file[file_name] += 1
    
    return {
        **current_metadata, # Inherit all base_metadata from the file
        "chunk_length": len(chunk.page_content), # Length of the actual chunk text
        "ingestion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # More precise timestamp
        "chunk_id": f"{file_name}_chunk_{chunk_counter_per_file[file_name]}", # Unique ID for this chunk
        "text_content_preview": chunk.page_content[:50] + "..." if len(chunk.page_content) > 50 else chunk.page_content # Small preview
    }