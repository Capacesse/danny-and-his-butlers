# src/feature_engineering.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import joblib
import os

def remove_urls(text):
    # Regex pattern for matching various types of URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Substitute all matched URLs with an empty string
    cleaned_text = url_pattern.sub('', text)
    return cleaned_text

def get_unified_data(df):
    '''
    :param file: this is the csv file that has been preprocessed
    :param review: if review is google then it can be more specific
    :return: a csv file that has all the new enginnered features such that the ML can now work on it
    '''

    # df = pd.read_csv(file)
    pd.set_option('display.max_columns', None)

    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["text"].apply(lambda x:analyzer.polarity_scores(x)["compound"])



    corpus = df["text"]

    #set stop words
    my_stop_words = [
        "menu",
        "taste",
        "eat",
        "ate",
        "order",
        "ordered",
        "sold",
        "staff",
    ]
    max_df_size = int(len(df.columns[0])*0.8)
    stop = text.ENGLISH_STOP_WORDS.union(my_stop_words)
    # Initialize TfidfVectorizer with the custom preprocessor
    vectorizer = TfidfVectorizer(preprocessor=remove_urls, min_df= 3, max_df = max_df_size, stop_words = list(stop))

    # Fit and transform the corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Define the path to save the vectorizer
    tfidf_vectorizer_path = '/content/drive/MyDrive/TikTok_Tech_Jam/models/tfidf_vectorizer.pkl'
    os.makedirs(os.path.dirname(tfidf_vectorizer_path), exist_ok=True)

    # Save the fitted vectorizer object
    joblib.dump(vectorizer, tfidf_vectorizer_path)

    print(f"TF-IDF Vectorizer saved to: {tfidf_vectorizer_path}")

    #transform to dataframe
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(),columns = vectorizer.get_feature_names_out())

    merged_df = pd.concat([df, df_tfidf], axis=1)

    # Load the pre-trained model.
    # The first time you run this, it will download the model files.
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Sentence Transformer model 'all-MiniLM-L6-v2' loaded.")

    """
    This model creates a 384-dimension vector for each piece of text, creating columns `embedding_dim_0` to `embedding_dim_383`.
    """

    ### Step 3: Generate the Embeddings
    """
    Feed the entire `text` column into the model. It will process all the reviews and output a matrix of numbers where each row is the vector for a review.
    """
    # Convert the 'text' column into a list of strings
    sentences = df['text'].tolist()

    print("\nGenerating embeddings for all reviews... (This may take a few minutes depending on dataset size)")

    # The .encode() method converts the text to numerical vectors
    # show_progress_bar=True gives you a helpful progress bar
    embeddings = model.encode(sentences, show_progress_bar=True)

    print(f"\nEmbeddings generated successfully. The shape of our new data is: {embeddings.shape}")

    ### Step 4: Create a New DataFrame for the Embeddings
    """
    The `embeddings` variable is currently a NumPy array. We need to convert it into a pandas DataFrame with the correct column names (`embedding_dim_0`, `embedding_dim_1`, etc.).
    """

    # Create the column names for our new DataFrame
    embedding_column_names = [f'embedding_dim_{i}' for i in range(embeddings.shape[1])]

    # Create the DataFrame
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_column_names)

    print("Created new DataFrame for the embedding features.")
    embeddings_df.head()

    ### Step 5: Combine with Your Main DataFrame
    """
    Finally, merge your original `processed_df` with the new `embeddings_df`. This gives you your final feature set.
    """

    # Reset indices to ensure a clean merge
    df.reset_index(drop=True, inplace=True)
    embeddings_df.reset_index(drop=True, inplace=True)

    # Combine the two DataFrames side-by-side
    all_features_df = pd.concat([merged_df, embeddings_df], axis=1)


    print(
        "Successfully merged embeddings with the main DataFrame to create an all-features set. This all-features set contains all features and text.")

    ### Step 6: Create two separate DataFrames - one for numerical features (model), one for lookup (policy)

    # Define columns for lookup vs. for modeling
    lookup_cols = ['text', 'business_name', 'author_name', 'photo', 'rating_category']
    model_cols = [col for col in all_features_df.columns if col not in lookup_cols]

    # Create the two separate DataFrames
    model_df = all_features_df[model_cols]
    lookup_df = all_features_df[lookup_cols]
    # model_df.to_csv('model_features.csv', index=False)
    # lookup_df.to_csv('reviews_lookup.csv', index=False)
    # all_features_df.to_csv("feature_engineered_reviews.csv", index=False)

    return model_df


def get_unified_data_for_inference(df):
    '''
    Preprocesses a new raw DataFrame for model inference.
    
    This function loads a pre-trained TF-IDF vectorizer to ensure the new data
    is processed with the exact same vocabulary as the training data.
    
    Args:
        df (pd.DataFrame): The new raw DataFrame to be processed. Must contain a 'text' column.
        tfidf_vectorizer_path (str): The file path to the saved TF-IDF vectorizer.
    
    Returns:
        pd.DataFrame: A DataFrame with all the required features for the ML model.
    '''
    tfidf_vectorizer_path = '/content/drive/MyDrive/TikTok_Tech_Jam/models/tfidf_vectorizer.pkl'

    def remove_urls(text):
        # Regex pattern for matching various types of URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        # Substitute all matched URLs with an empty string
        cleaned_text = url_pattern.sub('', text)
        return cleaned_text

    pd.set_option('display.max_columns', None)

    # --- Step 1: Preprocessing (same as training) ---
    nltk.download('vader_lexicon', quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    # Clean the text using the same function
    # df['text_clean'] = df['text'].apply(remove_urls)
    corpus = df["text"]
    
    # --- Step 2: Load and Transform with Saved Vectorizer ---
    print("Loading the saved TF-IDF vectorizer...")
    try:
        vectorizer = joblib.load(tfidf_vectorizer_path)
    except FileNotFoundError:
        print(f"Error: Vectorizer file not found at {tfidf_vectorizer_path}")
        return None

    print("Transforming new text data with the loaded vectorizer...")
    # Use the .transform() method, NOT .fit_transform()
    tfidf_matrix = vectorizer.transform(corpus)

    # Transform to DataFrame with correct column names from the loaded vectorizer
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    merged_df = pd.concat([df, df_tfidf], axis=1)

    # --- Step 3: Generate Embeddings (same as training) ---
    print("\nLoading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = df['text'].tolist()
    print("Generating embeddings for all reviews...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    embedding_column_names = [f'embedding_dim_{i}' for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_column_names)

    # --- Step 4: Combine all Features ---
    # Ensure all DataFrames have the same index for a clean merge
    df.reset_index(drop=True, inplace=True)
    # df_tfidf.reset_index(drop=True, inplace=True)
    embeddings_df.reset_index(drop=True, inplace=True)

    # Concatenate all features
    all_features_df = pd.concat([merged_df, embeddings_df], axis=1)

    # Return the unified model features
    model_cols = [col for col in all_features_df.columns if col not in ['text', 'business_name', 'author_name', 'photo', 'rating_category']]
    return all_features_df[model_cols]


