# src/image_analysis.py

def generate_image_descriptions_from_folder(image_folder_path: str) -> dict:
    """
    Walks a directory, finds all images, and generates a text description for each.

    This function initialises a pre-trained image-to-text model from Hugging Face
    and then iterates through all image files found in the specified folder
    and its subdirectories. It uses the model to generate a text description
    for each image and stores the results in a dictionary.

    Args:
        image_folder_path (str): The root directory to search for image files.

    Returns:
        dict: A dictionary where keys are image file paths and values are
              the generated text descriptions.
    """
    # 1. Initialise the image-to-text pipeline
    print("Loading Salesforce/blip-image-captioning-base model...")
    try:
        # Use a publicly available model like Salesforce/blip-image-captioning-base
        image_to_text_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        print("Image-to-text pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}

    # 2. Find all image files in the specified folder and its subdirectories
    image_files = []
    print(f"\nSearching for image files in: '{image_folder_path}'...")
    if not os.path.exists(image_folder_path):
        print(f"Error: The folder '{image_folder_path}' does not exist.")
        return {}

    for root, _, files in os.walk(image_folder_path):
        for file in files:
            # Check for common image file extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} image files.")

    # 3. Process each image to generate a text description
    image_descriptions = {}
    print("\nStarting image-to-text generation...")
    for image_path in image_files:
        try:
            # Open the image and ensure it's in RGB format for the model
            image = Image.open(image_path).convert("RGB")

            # Generate the text description
            # The pipeline returns a list of dictionaries, so we access the first one
            text_description = image_to_text_pipeline(image)[0]['generated_text']

            # Store the description in the dictionary
            image_descriptions[image_path] = text_description

            print(f"Processed: '{image_path}' -> '{text_description}'")
        except Exception as e:
            # Handle potential errors during image loading or model inference
            print(f"Error processing '{image_path}': {e}")
            image_descriptions[image_path] = f"Error: {e}"

    print("\nImage descriptions generated for all found files.")
    return image_descriptions
