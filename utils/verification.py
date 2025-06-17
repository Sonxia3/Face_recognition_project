from .preprocessing import img_to_encoding
import numpy as np
import tensorflow as tf
import os

def verify(identity, database, model):
    """
    Verifies whether the person in the most recent image matches the given identity from the database.

    Args:
    - identity (str): Name of the person to verify (must exist in the database).
    - database (dict): A dictionary mapping names to face embeddings.
    - model (Keras Model): Trained face recognition model.

    Returns:
    - dist (float): Similarity distance between the query image and the reference image.
    - door_open (bool): True if match (distance < 0.7), False otherwise.
    - query_path (str): Path to the image used for verification.
    """
    query_folder = "data/querry_images"
    images = [f for f in os.listdir(query_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        raise FileNotFoundError(f"No image found in {query_folder}")
    
    # Select the most recent image
    images.sort(key=lambda f: os.path.getmtime(os.path.join(query_folder, f)), reverse=True)
    query_path = os.path.join(query_folder, images[0]) 

    # Compute the embedding for the query image
    encoding = img_to_encoding(query_path, model)

    # Compute the distance to the target identity
    dist = np.linalg.norm(tf.subtract(encoding, database[identity]))

    # Decide if it's a match based on distance threshold
    if dist < 0.7:
        print(f"It's {identity}, welcome in!")
        door_open = True
    else:
        print(f"It's not {identity}, access denied.")
        door_open = False
        
    return dist, door_open, query_path
