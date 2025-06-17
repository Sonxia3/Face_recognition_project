from utils.database import (
    load_database_from_folder,
    add_person_to_database,
    remove_person_from_database,
    save_database,
    load_database_from_file
)
from utils.verification import verify
from utils.recognition import who_is_it
from utils.visualization import show_image, show_comparison
from utils.model_loader import load_fr_model
from utils.file_utils import find_reference_image

# Load the face recognition model
FRmodel = load_fr_model()

# Build a sample database from the reference_persons folder
database = load_database_from_folder("data/reference_persons", FRmodel)

# # Add or remove a person from the database (optional usage)
# add_person_to_database("myself_again", "data/reference_persons/unnamed.jpg", database, FRmodel)
# remove_person_from_database("myself_again", database)

# Save the database to a file (optional usage)
save_database(database, file_path="data/embeddings/database.pkl")

# # Load the database from file (optional usage)
database = load_database_from_file(file_path="data/embeddings/database.pkl")

# # Display a sample image (resized)
# show_image("data/reference_persons/unnamed.jpg", title="Image", size=(160, 160))


def run_verification(database, model):
    """
    Run identity verification using a query image and a specified person name.
    The most recent image in `data/querry_images` is used as input.
    """
    identity = input("Enter the name of the person to verify: ").strip()
    distance, is_match, query_path = verify(identity, database, model)
    reference_path = find_reference_image(identity)

    print(f"Distance: {distance:.4f} | Match: {'Yes' if is_match else 'No'}")
    show_comparison(query_path, reference_path, title1="Query image", title2="Reference image", size=(160, 160))


def run_recognition(database, model):
    """
    Run facial recognition to identify the person in the latest query image.
    The most similar identity from the database will be returned.
    """
    distance, name, query_path = who_is_it(database, model)
    reference_path = find_reference_image(name)

    print(f"Identified as: {name} | Distance: {distance:.4f}")
    show_comparison(query_path, reference_path, title1="Query image", title2="image of the identified person", size=(160, 160))


# Execute both functionalities
run_verification(database, FRmodel)
run_recognition(database, FRmodel)

