# 🧠 Face Recognition System

This project implements a face recognition system in Python using TensorFlow and Keras, based on the Inception-ResNet V1 model. It is designed with a modular architecture and offers three main features:

1. **Identity Verification**: Given an image, it checks if it matches a specific person in the database.  
2. **General Recognition**: Given an image, it identifies who the person is among the registered ones.  
3. **Visualization**: Allows plotting query images and comparing them with reference images.  

---

## 📁 Project Structure

face_recognition_project/  
├── main.py                        # Demo script  
├── model/  
│   └── inception_resnet_v1.py     # Model architecture  
├── models/  
│   └── pesos.h5                   # Trained weights (NOT included)  
├── data/  
│   ├── reference_images/          # Base images (known individuals)  
│   ├── query_images/              # New images to recognize  
│   └── embeddings/  
│       └── database.pkl           # Saved database (auto-generated)  
├── utils/  
│   ├── preprocessing.py           # Image loading and preprocessing  
│   ├── recognition.py             # Recognition logic  
│   ├── verification.py            # Verification logic  
│   ├── visualization.py           # Graphic functions  
│   ├── file_utils.py              # Helper functions  
│   ├── database.py                # Data loading, adding, and removing  
│   └── model_loader.py            # Load model with weights  
├── .gitignore  
├── requirements.txt  
└── README.md  

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download model weights  
To use the pretrained model, download the weights from the Releases section:

➡️ [Download pesos.h5](https://github.com/your-username/your-repo/releases)

Save it in the `models/` folder as `pesos.h5`.

3. Run the main script:
```bash
python main.py
```

---

### Main Functionalities

#### run_verification(database, model)  
→ Verifies whether a query image matches a specific person from the database.

Steps:  
1. Place the image to verify in `data/query_images/`  
2. Run `run_verification(database, model)`  
3. The system:  
   - Will ask for the name of the person to verify  
   - Selects the most recent image from `query_images`  
   - Compares that image with the reference of the given person  
   - Displays both images side by side and shows similarity distance  

**Returns:**  
- Person's name  
- Distance  
- Result: match or not  

#### run_recognition(database, model)  
→ Automatically identifies the most similar person in the database from a query image.

Steps:  
1. Place the image to identify in `data/query_images/`  
2. Run `run_recognition(database, model)`  
3. The system:  
   - Picks the most recent image  
   - Finds the closest match in the database  
   - Displays the query image alongside the closest reference  

**Returns:**  
- Identified name  
- Similarity distance  

---

## 🧪 Available Functions

### verify(identity, database, model)

- `verify(identity, database, model)`  
  → Verifies whether the most recent image in the query folder matches a specific person in the database.  

  To use this function:  
  1. Place the image to verify in `data/query_images/`  
  2. Call `verify(identity, database, model)`  
     - The system automatically selects the most recent image  
     - Compares it to the embedding of the given identity  

  **Returns:**  
  - `distance (float)`: similarity distance between query and reference  
  - `is_match (bool)`: True if match (distance < 0.7), False otherwise  
  - `query_path (str)`: path to the query image used  

- `who_is_it(database, model)`  
  → Identifies the most similar person from the database using a query image.  

  To use this function:  
  1. Place the image to identify in `data/query_images/`  
  2. Call `who_is_it(database, model)`  
     - The system selects the most recent image  
     - Searches for the closest embedding in the database  

  **Returns:**  
  - `min_dist (float)`: lowest distance between query and database images  
  - `identity (str)`: most similar person identified  
  - `query_path (str)`: path to the used query image  

- `load_database_from_folder(folder_path, model)`  
  → Creates an embedding dictionary from images in a folder.  

- `add_person_to_database(name, image_path, database, model)`  
  → Adds a new person to the system.  

- `remove_person_from_database(name, database)`  
  → Removes a person from the system.  

- `save_database(database)` / `load_database_from_file()`  
  → Saves and loads the embedding database.  

- `show_image(...)`, `show_comparison(...)`  
  → Functions for result visualization.  

---

## ⚠️ Usage Recommendations

To obtain the best results in face verification and recognition:

- **Manually crop the face in the image before uploading.**  
  The model expects a centered face occupying most of the image. Poor framing may reduce accuracy.  

- Use clear, well-lit images without obstructions (e.g., sunglasses, masks).  

- Images should be in .jpg, .png, or .jpeg format.  

---

## 📂 Sample Database

A small reference set is included in `data/reference_persons/` for testing the system.  
You can also add your own manually or using `add_person_to_database()`.  

---

## 📌 Notes

- The model uses 128-dimensional embeddings.  
- Images are resized to 160x160 pixels.  

---

## 🔗 Credits & Sources

- Architecture adapted from [FaceNet](https://arxiv.org/abs/1503.03832)  
- Model based on Inception-ResNet V1  
- Inspired by [facenet GitHub](https://github.com/davidsandberg/facenet)  

---
