import os
import pickle
from .preprocessing import img_to_encoding

def load_database_from_folder(folder_path, model):
    """
    Loads all face images from a folder and encodes them into a dictionary using the given model.

    Args:
    - folder_path (str): Path to the folder containing reference images.
    - model (Keras model): Face recognition model.

    Returns:
    - dict: A dictionary where keys are filenames (without extension) and values are embeddings.
    """
    database = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(filename)[0]
            path = os.path.join(folder_path, filename)
            database[name] = img_to_encoding(path, model)
    return database


def add_person_to_database(name, image_path, database, model):
    """
    Adds a person to the database by encoding their face image.

    Args:
    - name (str): Identifier for the person (used as the key in the dictionary).
    - image_path (str): Path to the person's image.
    - database (dict): Current dictionary of face embeddings.
    - model (Keras model): Face recognition model.

    Returns:
    - dict: Updated database dictionary.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    embedding = img_to_encoding(image_path, model)
    database[name] = embedding
    print(f"[INFO] Person '{name}' added to the database.")
    return database


def remove_person_from_database(name, database):
    """
    Removes a person from the database if they exist.

    Args:
    - name (str): Identifier of the person to remove.
    - database (dict): Current dictionary of face embeddings.

    Returns:
    - dict: Updated database dictionary.
    """
    if name in database:
        del database[name]
        print(f"[INFO] Person '{name}' removed from the database.")
    else:
        print(f"[WARN] Person '{name}' was not found in the database.")
    return database


def save_database(database, file_path="data/embeddings/database.pkl"):
    """
    Saves the current database dictionary to disk using pickle format.

    Args:
    - database (dict): The database to save.
    - file_path (str): Destination path for the pickle file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(database, f)
    print(f"[INFO] Database saved to {file_path}")


def load_database_from_file(file_path="data/embeddings/database.pkl"):
    """
    Loads a database dictionary from a pickle file.

    Args:
    - file_path (str): Path to the pickle file.

    Returns:
    - dict: Loaded database dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as f:
        database = pickle.load(f)
    print(f"[INFO] Database loaded from {file_path}")
    return database
