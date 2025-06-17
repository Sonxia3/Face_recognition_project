import glob
import os

def find_reference_image(identity, folder="data/reference_persons"):
    # Busca cualquier archivo con el nombre correspondiente, sin importar extensión
    pattern = os.path.join(folder, f"{identity}.*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No reference image found for '{identity}' in {folder}")
    return matches[0]  # Devuelve el primer archivo encontrado
