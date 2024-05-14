import os
import json
import torch
import logging
import sys
from io import StringIO
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

language_transformer_path = os.getenv("LANGUAGE_TRANSFORMER_PATH", os.path.join(os.path.expanduser("~"), ".cache", "transformers"))

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device for translation selected: {device}")

if device.type == 'cuda':
    logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
else:
    logger.info("CUDA is not available. Using CPU instead.")

model_name = "facebook/m2m100_1.2B"  # You can also use larger models for better accuracy
logger.info(f"Translation model: {model_name}")

# Load the tokenizer and model once, globally
tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=language_transformer_path)
model = M2M100ForConditionalGeneration.from_pretrained(model_name, cache_dir=language_transformer_path).to(device)

# Use FP16 precision if CUDA is available
if device.type == 'cuda':
    model = model.half()

def translate_to_german(text, src_lang):
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