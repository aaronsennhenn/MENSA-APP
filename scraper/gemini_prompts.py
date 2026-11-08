"""
These functions use the Google Gemini API to generate e.g. descriptions, embeddings or recipes from input dish names.
They are solely used in the run_cleanfilter.py script, which itself is regularly executed using a cronjob.
"""

import os
import re
import json
import time
import threading
import nltk
from nltk.corpus import stopwords
from secret import GEMINI_KEY

# Import the NEW google-genai SDK
from google import genai
from google.genai import types

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('german'))

# Setup Gemini Client using the new SDK
client = genai.Client(api_key=GEMINI_KEY, http_options={'api_version': 'v1beta'})

# Use a valid model name (1.5-flash is currently the standard for fast text tasks)
MODEL_NAME = 'gemini-3.5-flash-lite'

# The free-tier generate-content limit reported by Gemini is 15 requests/minute.
_MIN_SECONDS_BETWEEN_REQUESTS = 5.0
_last_request_at = 0.0
_request_lock = threading.Lock()


def _wait_for_request_slot():
    """Keep requests from this process at or below roughly 12 per minute."""
    global _last_request_at
    with _request_lock:
        now = time.monotonic()
        wait_seconds = _MIN_SECONDS_BETWEEN_REQUESTS - (now - _last_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _last_request_at = time.monotonic()


# Utility function to clean text
def preprocess_text(text):
    """
    Proprocesses text and removes stopword. Neccesary for embedding extraction.

    Parameters:
        text: Input string

    Returns:
        text.strip(): The processed text
    """
    text = re.sub(r"\[.*?\]", "", text).strip() # remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zäöüß\s]", "", text)
    text = text.replace("beilage", "").replace("wahl","").replace("mischsalat","").replace("blattsalat","").replace("bunter","")
    text = " ".join(text.split())

    # remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = " ".join(filtered_words)

    return text.strip()


# Function to extract ingredients
def ingredient_extraction(df, column):
    """
    Makes Gemini API call to generate a list of ingredients based on the dish name.
    """
    ingredients_de_list = []
    ingredients_en_list = []
    tokens_list = []
        
    for menu in df[column]:
        try:
            if not isinstance(menu, str) or not menu.strip():
                ingredients_de_list.append(None)
                ingredients_en_list.append(None)
                tokens_list.append(0)
                continue

            cleaned_text = preprocess_text(menu)
            
            # Send requests
            _wait_for_request_slot()
            response_de = client.models.generate_content(
                model=MODEL_NAME,
                contents=cleaned_text,
                config=types.GenerateContentConfig(
                    system_instruction="Extrahiere die Zutaten aus der folgenden Gerichtsbeschreibung. Berücksichtigen Sie nur Zutaten, die im Text erwähnt werden. Alle Zutaten sollten Substantive und auf Deutsch sein. Geben Sie die Zutaten in Form einer durch Kommata getrennten Liste an.",
                    temperature=0.2, 
                    max_output_tokens=200
                )
            )
            
            _wait_for_request_slot()
            response_en = client.models.generate_content(
                model=MODEL_NAME,
                contents=cleaned_text,
                config=types.GenerateContentConfig(
                    system_instruction="Extract the ingredients from the following dish description. Only consider ingredients that are mentioned in the string. All ingredients should be nouns and in English. Provide the ingredients as a list separated by commas.",
                    temperature=0.2, 
                    max_output_tokens=200
                )
            )
            
            # Extract ingredients and tokens used
            ingredients_de = response_de.text.strip()
            ingredients_en = response_en.text.strip()
            
            tokens_used = 0
            if response_de.usage_metadata:
                tokens_used += response_de.usage_metadata.total_token_count
            if response_en.usage_metadata:
                tokens_used += response_en.usage_metadata.total_token_count
            
            # Append to lists
            ingredients_de_list.append(ingredients_de)
            ingredients_en_list.append(ingredients_en)
            tokens_list.append(tokens_used)

        except Exception as e:
            print(f"Error in ingredients_extraction for menu item '{menu}': {e}")
            ingredients_de_list.append(None)
            ingredients_en_list.append(None)
            tokens_list.append(0)
    
    # Add columns to DataFrame
    df['ingredients_de'] = ingredients_de_list
    df['ingredients_en'] = ingredients_en_list
    
    # Handle tokens
    if 'tokens_used' in df.columns:
        df['tokens_used'] = df['tokens_used'] + tokens_list
    else:
        df['tokens_used'] = tokens_list
        
    return df


def embedding_extraction(df, column):
    """
    Makes Gemini API call to generate embeddings based on the dish name.
    """  
    
    gpt_embedding_list = []  

    for menu in df[column]:
        try:
            if not isinstance(menu, str) or not menu.strip():
                gpt_embedding_list.append(None)
                continue

            cleaned_text = preprocess_text(menu)

            # Get embedding for each meal
            _wait_for_request_slot()
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=cleaned_text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )

            # Extract list of floats from the new SDK response object
            embedding_values = response.embeddings[0].values
            embedding_str = json.dumps(embedding_values)
            gpt_embedding_list.append(embedding_str)
            
        except Exception as e: 
            print(f"Error extracting embeddings for menu item '{menu}': {e}")
            gpt_embedding_list.append(None)

    # Add columns to DataFrame
    df['gpt_embedding'] = gpt_embedding_list

    return df


def generate_description(df, column, menuLine):
    """
    Makes Gemini API call to generate descriptions based on the dish name.
    """   
    
    description_en_list = []
    description_de_list = []
    tokens_list = []

    # Define the dish filter
    dish_filter = ['Salat-/ Gemüsebuffet 100g', 'Beilagen vorport.', 'Dessert vorport.', 'Dessert SB', 'Beilagen SB']
        
    for dish, line in zip(df[column], df[menuLine]):
        try:
            if not isinstance(dish, str) or not dish.strip() or line in dish_filter:
                description_en_list.append("No description available.")
                description_de_list.append("Keine Beschreibung verfügbar.")
                tokens_list.append(0)
                continue

            # Request for German description
            _wait_for_request_slot()
            response_de = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"Beschreibe das Gericht: '{dish}'",
                config=types.GenerateContentConfig(
                    system_instruction="Du bist ein Student. Gib eine kurze Beschreibung des Gerichts auf Deutsch. Verwende einen prägnanten und einfachen Ton. Beispiel für das Gericht Chili Con Carne: Ein herzhaftes, würziges Gericht aus Rinderhackfleisch, roten Bohnen, Tomaten, Zwiebeln, Knoblauch und Chili. Gewürzt mit Paprika, Kreuzkümmel und Chili für Schärfe und Geschmack.",
                    temperature=0.01, 
                    max_output_tokens=300
                )
            )

            description_de = response_de.text.strip()
            tokens_used = response_de.usage_metadata.total_token_count if response_de.usage_metadata else 0

            # Request for English Translation
            _wait_for_request_slot()
            translation_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"Translate this description:\n\n{description_de}",
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert translator. Please translate the following German description into clear, natural English. Maintain the format and tone, and ensure that the description is easy to understand.",
                    temperature=0.2, 
                    max_output_tokens=100
                )
            )

            description_en = translation_response.text.strip()
            if translation_response.usage_metadata:
                tokens_used += translation_response.usage_metadata.total_token_count

            # Append results
            description_en_list.append(description_en)
            description_de_list.append(description_de)
            tokens_list.append(tokens_used)

        except Exception as e:
            print(f"Error in generate_description for menu item '{dish}': {e}")
            description_en_list.append("No description available.")
            description_de_list.append("Keine Beschreibung verfügbar.")
            tokens_list.append(0)

    # Add results to DataFrame
    df['description_en'] = description_en_list
    df['description_de'] = description_de_list

    # Handle tokens
    if 'tokens_used' in df.columns:
        df['tokens_used'] = df['tokens_used'] + tokens_list
    else:
        df['tokens_used'] = tokens_list

    return df


def classify_dish_taste(df, column):
    """
    Makes Gemini API call to classify the taste of a dish based on the dish name.
    """ 
    taste_en_list = []
    taste_de_list = []
    tokens_list = []

    for menu in df[column]:
        try:
            if not isinstance(menu, str) or not menu.strip():
                taste_en_list.append(None)
                taste_de_list.append(None)
                tokens_list.append(0)
                continue

            _wait_for_request_slot()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"Classify the dish: '{menu}'",
                config=types.GenerateContentConfig(
                    system_instruction="Classify the dish into one of the following categories: Fettig, Leicht, or Süß. Respond with only one word.",
                    temperature=0.2, 
                    max_output_tokens=10
                )
            )

            taste_de = response.text.strip()
            tokens_used = response.usage_metadata.total_token_count if response.usage_metadata else 0
            
            taste_en = None
            if taste_de == "Fettig":
                taste_en = "Fatty"
            elif taste_de == "Leicht":
                taste_en = "Light"
            elif taste_de == "Süß":
                taste_en = "Sweet"
                
            taste_de_list.append(taste_de)
            taste_en_list.append(taste_en)
            tokens_list.append(tokens_used)

        except Exception as e:
            print(f"Error in classify_dish_taste for menu item '{menu}': {e}")
            taste_en_list.append(None)
            taste_de_list.append(None)
            tokens_list.append(0)

    df['taste_de'] = taste_de_list
    df['taste_en'] = taste_en_list

    if 'tokens_used' in df.columns:
        df['tokens_used'] = df['tokens_used'] + tokens_list
    else:
        df['tokens_used'] = tokens_list

    return df


def generate_recipe(df, column, menuLine):
    """
    This function generates recipes in english and german language for a given dish.
    """   
        
    recipe_en_list = []
    recipe_de_list = []
    tokens_list = []

    dish_filter = ['Salat-/ Gemüsebuffet 100g', 'Beilagen vorport.', 'Dessert vorport.', 'Dessert SB', 'Beilagen SB']
        
    for dish, line in zip(df[column], df[menuLine]):
        try:
            if not isinstance(dish, str) or not dish.strip() or line in dish_filter:
                recipe_en_list.append("No recipe available.")
                recipe_de_list.append("Kein Rezept verfügbar.")
                tokens_list.append(0)
                continue

            _wait_for_request_slot()
            response_de = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"Bitte geben Sie ein kurzes Rezept an für das Gericht: '{dish}'.",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "Sie sind ein Profikoch. Erstellen Sie ein kurzes und einfaches Rezept für das folgende Gericht. "
                        "Das Rezept sollte Zutaten und eine Schritt-für-Schritt-Anleitung enthalten, in deutscher Sprache verfasst und für einen Hausmann geeignet sein. "
                        "Formatieren Sie das Rezept wie folgt: \n"
                        "Zutaten:\n- [Zutat 1]\n- [Zutat 2]\n\nAnleitung:\n1. [Schritt 1]\n2. [Schritt 2] \n"
                        "Stellen Sie sicher, dass die Zutaten und Anweisungen vollständig und klar sind, ohne mitten im Satz abzubrechen. "
                        "Vermeiden Sie abschließende Bemerkungen wie 'Guten Appetit'. Geben Sie nur das Rezept an."
                    ),
                    temperature=0.1, 
                    max_output_tokens=500
                )
            )

            recipe_de = response_de.text.strip()
            tokens_used = response_de.usage_metadata.total_token_count if response_de.usage_metadata else 0

            _wait_for_request_slot()
            translation_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"Translate this recipe:\n\n{recipe_de}",
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert translator. Please translate the following German recipe into clear, natural English. Maintain the format and tone, and ensure that the instructions are easy to follow.",
                    temperature=0.1, 
                    max_output_tokens=500
                )
            )

            recipe_en = translation_response.text.strip()
            if translation_response.usage_metadata:
                tokens_used += translation_response.usage_metadata.total_token_count

            recipe_en_list.append(recipe_en)
            recipe_de_list.append(recipe_de)
            tokens_list.append(tokens_used)

        except Exception as e:
            print(f"Error in generate_recipe for menu item '{dish}': {e}")
            recipe_en_list.append("No recipe available.")
            recipe_de_list.append("Kein Rezept verfügbar.")
            tokens_list.append(0)
    
    df['recipe_en'] = recipe_en_list
    df['recipe_de'] = recipe_de_list

    if 'tokens_used' in df.columns:
        df['tokens_used'] = df['tokens_used'] + tokens_list
    else:
        df['tokens_used'] = tokens_list
        
    return df


def classify_missing_filters(dish_name):
    """
    Utilizes Gemini API. This function classifies the dish into the predefined filter categories.
    """
    
    _wait_for_request_slot()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=dish_name,
        config=types.GenerateContentConfig(
            system_instruction="The string that I provide is a dish name. Please classify the dish into one of the following categories: vegan, vegetarian, fish, poultry, veal, lamb, beef, pork, game. Only return the classification string",
            temperature=0.2, 
            max_output_tokens=200
        )
    )

    result = response.text.strip().split("\n")
    return result[0].strip() if len(result) > 0 else ""
