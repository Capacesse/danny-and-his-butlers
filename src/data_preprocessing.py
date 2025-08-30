# src/data_preprocessing.py

import pandas as pd
import numpy as np
import re
import os
from PIL import Image
from transformers import pipeline

# Note: All necessary imports are at the top of the file.

def preprocess_data(input_file: str, output_file: str, image_folder_path: str) -> pd.DataFrame:
    """
    Performs comprehensive data preprocessing and feature engineering on a review dataset.

    This script takes a raw CSV, line-delimited JSON, or Excel file, cleans the
    review text, extracts a set of metadata-based features, and saves the
    result to a new CSV file.

    Args:
        input_file (str): The path to the raw input file. Supported formats are
                          .csv, .json, .xlsx, and .xls.
        output_file (str): The path where the preprocessed CSV file will be saved.
        image_folder_path (str): The path to the directory containing the images.

    Returns:
        pd.DataFrame: The final preprocessed DataFrame.
    """

    # 1. Load the raw data based on file type
    try:
        file_extension = os.path.splitext(input_file)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(input_file)
        elif file_extension == '.json':
            df = pd.read_json(input_file, lines=True)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
        else:
            print(f"Error: Unsupported file type '{file_extension}'. This script supports .csv, .json, .xlsx, and .xls.")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please ensure it's in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return pd.DataFrame()

    # Drop any duplicate rows before processing
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Handle missing values and standardize column names
    column_mapping = {
        'pics': 'photo',
        'name': 'author_name',
        'reviewer_name': 'author_name',
        'user_name': 'author_name',
        'biz_name': 'business_name',
        'gmap_id': 'business_name',
        'business': 'business_name',
        'stars': 'rating',
        'rating_category': 'rating_category'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Fill NaN values to prevent errors during string operations
    for col in df.columns:
      if df[col].dtype == 'object':
        df[col].fillna('', inplace=True)
      elif pd.api.types.is_numeric_dtype(df[col]):
        # Use -1 as a specific placeholder for all missing numeric values
        df[col].fillna(-1, inplace=True)
      else:
        df[col].fillna(-1, inplace=True)

    # 3. Feature Extraction (on original, raw text)
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|www\.[^\s/$.?#]+\.[^\s/$.?#]+|[\w\.-]+@[\w\.-]+|\b(http|https|ftp|ftps)\b'
    df['has_url'] = df['text'].apply(lambda x: 1 if pd.notna(x) and re.search(url_pattern, str(x), re.IGNORECASE) else 0)

    df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!') if pd.notna(x) else 0)
    df['question_mark_count'] = df['text'].apply(lambda x: str(x).count('?') if pd.notna(x) else 0)
    # Ellipsis definition is two or more periods
    df['ellipsis_count'] = df['text'].apply(
        lambda x: len(re.findall(r'\.{2,}', str(x))) if pd.notna(x) else 0
    )
    # is_zero_visit uses word boundaries
    zero_visit_keywords = r'\b(0|zero|never visited|never been|haven\'t been|have not been|didn\'t visit|did not visit|never actually|never tried|just by looking|judging by|i can already tell|never ordered|never actually ordered|from what i\'ve heard|from whati\'ve read|from what people told me)\b'
    df['is_zero_visit'] = df['text'].apply(
        lambda x: 1 if pd.notna(x) and re.search(zero_visit_keywords, str(x).lower()) else 0
    )
    df['all_caps_word_count'] = df['text'].apply(
        lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 1)
    )
    df['capital_letter_percentage'] = df['text'].apply(
        lambda x: (sum(1 for char in str(x) if char.isupper()) / len(str(x))) * 100 if len(str(x)) > 0 else 0
    )

    # 4. Standardize ratings to a 1-5 scale
    if 'rating' in df.columns and not df['rating'].empty and pd.api.types.is_numeric_dtype(df['rating']):
        max_rating = df['rating'].max()
        # Only standardize if max rating is greater than 5 and is a finite number
        if max_rating > 5 and pd.notna(max_rating) and max_rating > 0:
            df['rating'] = df['rating'].apply(lambda x: round((x / max_rating) * 5) if pd.notna(x) and x > 0 else -1)
        # Ensure ratings are at least 1, unless the original was 0 or a placeholder
        df['rating'] = df['rating'].apply(lambda x: max(1, x) if x > 0 else -1)

    # 5. Clean the initial review text
    # Convert to lowercase
    df['cleaned_initial_text'] = df['text'].astype(str).str.lower()
    # Remove URL or emails
    df['cleaned_initial_text'] = df['cleaned_initial_text'].apply(lambda x: re.sub(url_pattern, '', str(x), flags=re.IGNORECASE)
    )
    # Deal with numbers
    df['cleaned_initial_text'] = df['cleaned_initial_text'].apply(lambda x: re.sub(r'\b\d+\b', '<NUM>', str(x))
    )
    # Remove special characters and punctuation
    df['cleaned_initial_text'] = df['cleaned_initial_text'].apply(lambda x: re.sub(r'[^a-z0-9\s<>]', '', str(x))
    )
    # Tidy up whitespace
    df['cleaned_initial_text'] = df['cleaned_initial_text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip()
    )

    # 6. Find word_count and char_count from this initial cleaned text
    df['word_count'] = df['cleaned_initial_text'].apply(lambda x: len(x.split()) if len(x.strip()) > 0 else 0)
    df['char_count'] = df['cleaned_initial_text'].str.len()

    # 7. Generate and clean image descriptions
    print("\nLoading Salesforce/blip-image-captioning-base model...")
    try:
        # Use a publicly available model like Salesforce/blip-image-captioning-base
        image_to_text_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        print("Image-to-text pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        image_to_text_pipeline = None

    def get_image_description(photo_path, image_folder):
        """Helper function to process a single photo path."""
        # Join the base folder and the photo path to get the full file path
        full_path = os.path.join(image_folder, photo_path)
        if not full_path or not os.path.exists(full_path) or image_to_text_pipeline is None:
            return ""
        try:
            image = Image.open(full_path).convert("RGB")
            # Generate description and clean it immediately
            description = image_to_text_pipeline(image)[0]['generated_text']
            description = re.sub(r'[^a-z0-9\s<>]', '', description.lower()).strip()
            return f"a photo shows {description}. "
        except Exception as e:
            print(f"Error processing image {full_path}: {e}")
            return ""

    # Apply the function to the 'photo' column to create the new cleaned_image_description column
    # The .apply method handles each row, including cases where 'photo' might be a list
    df['cleaned_image_description'] = df['photo'].apply(
        lambda p: get_image_description(p[0], image_folder_path) if isinstance(p, list) and p else get_image_description(p, image_folder_path)
    )

    # 8. Merge the cleaned text with the cleaned image descriptions
    df['unified_text'] = df['cleaned_initial_text'].fillna('') + ' ' + df['cleaned_image_description'].fillna('')
    df['unified_text'] = df['unified_text'].str.strip()

    # 9. Finalize DataFrame preparation
    final_df = df.drop(columns=['text', 'cleaned_initial_text', 'cleaned_image_description'])
    final_df.rename(columns={'unified_text': 'text'}, inplace=True)

    # Convert data types to match the specified contract
    final_df['has_url'] = final_df['has_url'].astype('int64')
    final_df['is_zero_visit'] = final_df['is_zero_visit'].astype('int64')
    final_df['word_count'] = final_df['word_count'].astype('int64')
    final_df['char_count'] = final_df['char_count'].astype('int64')
    final_df['exclamation_count'] = final_df['exclamation_count'].astype('int64')
    final_df['question_mark_count'] = final_df['question_mark_count'].astype('int64')
    final_df['ellipsis_count'] = final_df['ellipsis_count'].astype('int64')
    final_df['all_caps_word_count'] = final_df['all_caps_word_count'].astype('int64')
    final_df['rating'] = final_df['rating'].astype('int64')

    if 'rating_category' in final_df.columns:
        final_df['rating_category'] = final_df['rating_category'].astype('category')
    for col in final_df.columns:
        if col in ['text', 'photo', 'author_name', 'business_name']:
            final_df[col] = final_df[col].astype('string')
        elif col in ['has_url', 'is_zero_visit', 'word_count', 'char_count', 'exclamation_count', 'question_mark_count', 'ellipsis_count', 'all_caps_word_count', 'rating']:
            final_df[col] = final_df[col].astype('int64')
        elif col == 'capital_letter_percentage':
            final_df[col] = final_df[col].astype('float64')
        elif col == 'rating_category' and 'rating_category' in final_df.columns: # one category is string and other is boolean (if data has rating_category or not) -> it is combined here to form one 'category' column
            final_df[col] = final_df[col].astype('category')

    final_df.to_csv(output_file, index=False)
    print(f"Data preprocessing complete. Saved to '{output_file}' with {len(final_df)} rows.")

    return final_df
