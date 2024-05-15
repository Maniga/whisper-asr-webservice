import os
import json
import re
import torch
import logging
import sys
from io import StringIO
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect, LangDetectException

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

def translate_to_german(text, src_lang=None):

    if src_lang is None:
        src_lang = detect_language(text)
        if src_lang is None:
            return json.dumps({"error": "Could not detect language"}, ensure_ascii=False)
    
    # Set the source and target languages
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = "de"

    logger.info(f"Translating from {src_lang} to de")

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Translate each sentence
    translations = []
    for sentence in sentences:
        encoded = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("de"))
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        translations.append(translation)
    
    # Combine all translations into a single string
    full_translation = ' '.join(translations)

    # Log the translation for debugging
    logger.info(f"Translation: {full_translation}")

    # Create a JSON object with the result
    result = {
        "source_language": src_lang,
        "target_language": "de",
        "original_text": text,
        "translated_text": full_translation
    }

    return json.dumps(result, ensure_ascii=False)

def detect_language(text):
    """
    Detects the language of the given text.
    
    Args:
    - text (str): The text to detect the language of.
    
    Returns:
    - str: The ISO 639-1 code of the detected language, or None if detection fails.
    """
    try:
        detected_lang = detect(text)
        logger.info(f"Detected language: {detected_lang}")
        return detected_lang
    except LangDetectException as e:
        logger.error(f"Language detection failed: {e}")
        return None