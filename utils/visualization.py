import tensorflow as tf
import matplotlib.pyplot as plt

def show_image(image_path, title="Resized image", size=(160, 160)):
    """
    Displays a single image with a title and no axes.

    Args:
    - image_path (str): Path to the image file.
    - title (str): Title displayed above the image (default: 'Resized image').
    - size (tuple): Target size to resize the image (default: 160x160).
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()
    

def show_comparison(query_path, reference_path, title1="Query image", title2="Reference image", size=(160, 160)):
    """
    Displays two images side by side for visual comparison.

    Args:
    - query_path (str): Path to the input/query image to be recognized.
    - reference_path (str): Path to the identified or reference image.
    - title1 (str): Title for the first image (default: 'Query image').
    - title2 (str): Title for the second image (default: 'Reference image').
    - size (tuple): Target size to resize both images (default: 160x160).
    """
    img_query = tf.keras.preprocessing.image.load_img(query_path, target_size=size)
    img_reference = tf.keras.preprocessing.image.load_img(reference_path, target_size=size)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_query)
    axs[0].axis("off")
    axs[0].set_title(title1)

    axs[1].imshow(img_reference)
    axs[1].axis("off")
    axs[1].set_title(title2)

    plt.show()
