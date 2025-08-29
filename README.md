# danny-and-his-butlers

## Project Overview

An end-to-end ML pipeline designed to restore trust and integrity to online reviews. It directly addresses the core problem of low-quality, irrelevant, and fraudulent reviews which mislead consumers, unfairly harm businesses, and create a massive manual moderation burden for platforms.

## Our Pipeline

Our solution uses a (TBC) approach:

## Data Processing & Feature Engineering

Our pipeline performs several steps to clean the data and extract meaningful features before modelling:
### Text-Based Features
The raw review text is processed to create a rich set of features that help identify spam and low-quality content, including:
-   `has_url`: A binary flag for reviews containing URLs.
-   `exclamation_count` / `question_mark_count`: To capture review tone.
-   `is_zero_visit`: A flag that detects keywords like "never been" or "heard it was bad" to identify non-visitors.
-   `all_caps_word_count`: To detect excessive use of capitalisation.

### Image-Based Features (Proof of Visit)
To validate that a review is from a genuine visitor, our pipeline analyses the attached photo:
1.  **Image Captioning:** We use the `Salesforce/blip-image-captioning-base` model to generate a text description for each review photo.
2.  **Irrelevant Image Filtering:** We then apply a rule-based filter to identify and flag reviews with irrelevant images (e.g., photos of menus, signs, or error messages), further cleaning our dataset.

## Tech Stack & Rationale

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
2. Ensure the required dataset is in the `/data` folder.
3. Run all cells from top to bottom to execute the full pipeline.

## Team Contributions

- **Project Lead & MLOps:** Julius Ng Hwee Chong
- **Data Engineer:** Phoa Li Lynn Sheryl
- **Feature Engineer & NLP Specialist:** Park Soohwan
- **Model Developer & Pseudo-Labeling Specialist:** Danny Lee Han Keat
- **Evaluation & Policy Engineer:** Mahesh Sanjana
