from .preprocessing import img_to_encoding
import tensorflow as tf
import os

def who_is_it(database, model):
    """
    Identifies the person in the most recent image inside the `data/querry_images` folder
    by comparing its face embedding to the ones stored in the database.

    Args:
    - database (dict): A dictionary mapping names to face embeddings.
    - model (Keras Model): Trained face recognition model.

    Returns:
    - min_dist (float): The shortest distance between the query image and any image in the database.
    - identity (str): The name of the most likely matching person.
    - query_path (str): The path to the image used for the query.
    """
    query_folder = "data/querry_images"
    images = [f for f in os.listdir(query_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        raise FileNotFoundError(f"No image found in {query_folder}")
    
    # Select the most recent image in the folder
    images.sort(key=lambda f: os.path.getmtime(os.path.join(query_folder, f)), reverse=True)
    query_path = os.path.join(query_folder, images[0])

    # Compute the embedding of the query image
    encoding = img_to_encoding(query_path, model)
    
    # Compare with all embeddings in the database
    min_dist = float("inf")
    identity = None

    for name, db_enc in database.items():
        dist = tf.linalg.norm(tf.subtract(encoding, db_enc)).numpy()
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 1.5:
        print("Not in the database.")
    else:
        print(f"It's {identity}, distance: {min_dist:.4f}")
    
    return min_dist, identity, query_path
