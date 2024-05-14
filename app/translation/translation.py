import os
import json
import torch
from io import StringIO
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

language_transformer_path = os.getenv("LANGUAGE_TRANSFORMER_PATH", os.path.join(os.path.expanduser("~"), ".cache", "transformers"))

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/m2m100_1.2B"  # You can also use larger models for better accuracy

def translate_to_german(text, src_lang):
    # Initialize the tokenizer and model on the specified device
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=language_transformer_path)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name, cache_dir=language_transformer_path).to(device)

    # Set the target language
    tokenizer.src_lang = src_lang  # Default to English, model will auto-detect if wrong
    tokenizer.tgt_lang = "de"

    # Encode the text and move tensors to the appropriate device
    encoded = tokenizer(text, return_tensors="pt").to(device)

    # Generate translation
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("de"))

    # Decode and return the translation
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

     # Create a JSON object with the result
    result = {
        "source_language": src_lang,
        "target_language": "de",
        "original_text": text,
        "translated_text": translation
    }

    return json.dumps(result, ensure_ascii=False)