# src/feature_engineering.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sentence_transformers import SentenceTransformer

def get_unified_data(file):
    '''
    :param file: this is the csv file that has been preprocessed
    :param review: if review is google then it can be more specific
    :return: a csv file that has all the new enginnered features such that the ML can now work on it
    '''

    def remove_urls(text):
        # Regex pattern for matching various types of URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        # Substitute all matched URLs with an empty string
        cleaned_text = url_pattern.sub('', text)
        return cleaned_text


    df = pd.read_csv(file)
    pd.set_option('display.max_columns', None)

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
    vectorizer = TfidfVectorizer(preprocessor=remove_urls, min_df= 5, max_df = max_df_size, stop_words = list(stop))

    # Fit and transform the corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)

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
    lookup_cols = ['text', 'business_name', 'author_name', 'photo']
    model_cols = [col for col in all_features_df.columns if col not in lookup_cols]

    # Create the two separate DataFrames
    model_df = all_features_df[model_cols]
    lookup_df = all_features_df[lookup_cols]
    model_df.to_csv('model_features.csv', index=False)
    lookup_df.to_csv('reviews_lookup.csv', index=False)
    all_features_df.to_csv("feature_engineered_reviews.csv", index=False)

    return all_features_df