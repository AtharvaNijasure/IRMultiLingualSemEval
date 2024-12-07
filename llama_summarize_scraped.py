import transformers
from huggingface_hub import login
import torch
from utils import read_json
import json
from tqdm import tqdm
import csv
import re

def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def clean_string(input_string):
    # Split the string by whitespace and rejoin with a single space
    text = input_string.replace('\n', ' ')
    return normalize_spaces(text)
    

def llama31_generate_batch(pipeline, prompts, model_id, system=None, batch_size=10, ids_=None, output_file="outputs_summarized.csv"):
    """
    Generate outputs in batches and save them to a CSV file after each batch.
    
    Args:
        pipeline: The Hugging Face generation pipeline.
        prompts: List of prompts to generate responses for.
        model_id: Model identifier (optional, not used directly here).
        system: System message content (optional).
        batch_size: Number of prompts to process in each batch.
        ids_: List of IDs corresponding to the prompts.
        output_file: Path to the output CSV file.
    """
    # Open the CSV file and write the header if it doesn't exist
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Generated Output"])

    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_ids = ids_[i:i+batch_size]
        batch_messages = []

        # Prepare batch messages
        for prompt in batch_prompts:
            if system is not None:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt}
                ]
            batch_messages.append(messages)

        # Generate responses for the batch
        outputs = pipeline(
            batch_messages,
            max_new_tokens=100,
        )

        # Extract and save generated outputs
        with open(output_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for idx, output in enumerate(outputs):
                generated_text = output[0]["generated_text"][-1]["content"].strip()
                writer.writerow([batch_ids[idx], generated_text])
                print(f"Saved Output for ID {batch_ids[idx]}")

if __name__ == "__main__":
    # Example usage
    
    file_path = "scraped_content.jsonl"
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as JSON
            ex = json.loads(line)
            data.append(ex)

    scraped_text = [clean_string((data[i]["content"]).strip()) for i in range(len(data))]
    ids_ = [data[i]["fact_check_id"] for i in range(len(data))]
    
    print(len(scraped_text))
    print(len(ids_))
    filtered_data = [(text, ids_[i]) for i, text in enumerate(scraped_text) if text != ""]
    scraped_text, ids_ = zip(*filtered_data) if filtered_data else ([], [])
    # Convert back to lists
    scraped_text = list(scraped_text)
    ids_ = list(ids_)
    print(len(scraped_text))
    print(len(ids_))
    
    inst = "Summarize the content of the text provided in short.\n"
    
    prompts = [inst + "Text:\n" + scraped_text[i] for i in range(len(scraped_text))]  # List of prompts to process
    print(prompts[0])
   
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Specify your model ID
    huggingface_token = "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"  # Set Hugging Face token

    # Login to Hugging Face
    login(token=huggingface_token)

    # Set up pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Generate outputs in batches and save to CSV
    llama31_generate_batch(pipeline, prompts, model_id, system=None, batch_size=8, ids_=ids_, output_file="generated_outputs_summarized.csv")
