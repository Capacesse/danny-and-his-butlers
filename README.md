# danny-and-his-butlers

## Project Overview

An end-to-end ML pipeline designed to restore trust and integrity to online reviews. It directly addresses the core problem of low-quality, irrelevant, and fraudulent reviews which mislead consumers, unfairly harm businesses, and create a massive manual moderation burden for platforms.

## Our Pipeline
Our solution is a multi-stage pipeline that uses a combination of rule-based logic and machine learning to achieve high accuracy.

1.  **Image Analysis & Preprocessing:** The pipeline begins by analysing the review's photo using a BLIP model to generate a text caption. This caption is merged with the original, cleaned review text to create a single, unified text field for analysis.

2.  **Advanced Feature Engineering:** We convert the unified text into a rich numerical representation using two parallel techniques:
    * **TF-IDF Vectors:** To capture the importance of specific keywords.
    * **Sentence Embeddings:** To capture the overall contextual meaning and sentiment.

3.  **Two-Stage Pseudo-Labelling:** To overcome the lack of training data, we employ a two-stage labelling strategy:
    * **Stage 1 (Rule-Based):** High-confidence rules are applied to the dataset to accurately label the most obvious cases (e.g., clear advertisements, valid reviews).
    * **Stage 2 (Baseline Model):** A baseline `LogisticRegression` model is trained on these high-confidence labels. This model then predicts labels for the remaining, more ambiguous reviews in the dataset.

4.  **Final Model Training:** A powerful **Stacking Ensemble Model** is trained on the fully labeled dataset. This final model combines the strengths of three specialised base models and a final meta-model to make the definitive classification.

## Tech Stack & Rationale
-   **Hugging Face Transformers & PyTorch:** The backbone of our project, providing access to state-of-the-art models for image captioning (BLIP) and text understanding.
-   **Scikit-learn & LightGBM:** Used to build our powerful stacking ensemble model, combining the reliability of logistic regression with the high performance of LightGBM.
-   **Sentence-Transformers:** Employed to generate high-quality semantic embeddings, allowing our model to understand the nuanced meaning of reviews.
-   **Pandas:** The primary tool for all data manipulation and management throughout the pipeline.
-   **Joblib:** Used for saving and loading our final trained model pipeline.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone [https://github.com/Capacesse/danny-and-his-butlers.git](https://github.com/Capacesse/danny-and-his-butlers.git)
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your Hugging Face token in your Google Colab Secrets manager under the name `TikTokTechJam2025`.

## How to Reproduce Results
1. Open the `/notebooks/Main_Pipeline.ipynb` notebook in Google Colab.
2. Ensure the required dataset (`merged_reviews_4.csv` and the `dataset` folder) is in the `/data` folder within your project's Google Drive directory.  
   - `dataset` folder should have the following structure:  
`dataset`  
|- `indoor_atmosphere`  
|- `menu`  
|- `outdoor_atmosphere`  
|- `taste`  
   
3. Run all cells from top to bottom to execute the full training and evaluation pipeline.
4. For your own testing, you must have your own dataset in the `\data` folder as well

## Team Contributions
- **Project Lead & Pipelining:** Julius Ng Hwee Chong
- **Data Preprocessing:** Phoa Li Lynn Sheryl
- **Feature Engineer:** Park Soohwan
- **Model Developer & Pseudo-Labelling:** Danny Lee Han Keat
- **Policy & Evaluation:** Mahesh Sanjana
