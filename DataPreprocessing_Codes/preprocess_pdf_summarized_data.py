import os

def process_text_file(filepath):
    """
    Reads a single text file, cleans it by removing blank lines and
    newline characters, and returns all text as a single line with spaces.
    """
    processed_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 1. Remove leading/trailing whitespace from the line
                stripped_line = line.strip()
                # 2. Skip blank lines
                if stripped_line:
                    processed_lines.append(stripped_line)
        
        # 3. Join all processed lines into a single string, separated by a space
        # This effectively removes original newlines and inserts a space instead
        return " ".join(processed_lines)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

def process_directory(directory_path, output_directory=None):
    """
    Processes all text files in a given directory, applies cleaning,
    and optionally saves the processed content to an output directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    print(f"Processing files in: {directory_path}")
    
    processed_documents = {}

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        
        # Ensure it's a file and not a directory, and optionally check extension
        if os.path.isfile(filepath) and filename.lower().endswith(('.txt', '.md')): # You can add more extensions here
            print(f"  Processing file: {filename}")
            cleaned_text = process_text_file(filepath)
            
            if cleaned_text is not None:
                processed_documents[filename] = cleaned_text
                
                if output_directory:
                    output_filepath = os.path.join(output_directory, filename)
                    try:
                        with open(output_filepath, 'w', encoding='utf-8') as outfile:
                            outfile.write(cleaned_text)
                        print(f"    Saved processed content to: {output_filepath}")
                    except Exception as e:
                        print(f"    Error saving processed file {output_filepath}: {e}")
            else:
                print(f"  Skipped file {filename} due to an error.")
        else:
            print(f"  Skipping non-text file or directory: {filename}")

    return processed_documents

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    input_folder = "./data/Summarized_PDFs/" # Your folder with original text files
    output_folder = "./data/CleanedPDFsSummarized/" # Folder to save cleaned files

    # Process the files
    cleaned_data = process_directory(input_folder, output_folder)
    
    for filename, content in cleaned_data.items():
        print(f"File: {filename}\nContent: {content[:200]}...")
        print("-" * 40)

    print("\nProcessing complete.")