"""
predecir.py
------------
Carga el modelo ya entrenado (modelo_semaforos.h5) y predice la clase
de una o varias imágenes nuevas, sin necesidad de volver a entrenar.

Uso - una sola imagen:
    python predecir.py --imagen ruta/a/foto.jpg

Uso - todas las imágenes de una carpeta:
    python predecir.py --carpeta ruta/a/carpeta_con_fotos

Requisitos: el mismo entorno donde entrenaste (tensorflow, numpy),
y que 'dataset/training_set' siga existiendo al mismo nivel desde
donde corres este script (se usa solo para saber el orden de las clases,
no vuelve a entrenar nada).
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODELO_PATH = "modelo_semaforos.h5"
TRAINING_DIR = "dataset/training_set"
TARGET_SIZE = (64, 64)
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png"}


def obtener_orden_clases(training_dir: str):
    """
    Keras asigna los índices de clase en orden alfabético de las subcarpetas
    dentro de training_set. Replicamos ese mismo orden aquí para poder
    interpretar correctamente la salida del modelo sin tener que recrear
    el ImageDataGenerator.
    """
    carpetas = sorted([d.name for d in Path(training_dir).iterdir() if d.is_dir()])
    if not carpetas:
        raise RuntimeError(f"No se encontraron subcarpetas de clases en '{training_dir}'")
    return carpetas


def predecir_imagen(modelo, ruta_imagen: Path, clases: list):
    img = image.load_img(str(ruta_imagen), target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # mismo reescalado que en el entrenamiento

    resultado = modelo.predict(img_array, verbose=0)[0]
    indice_predicho = int(np.argmax(resultado))
    clase_predicha = clases[indice_predicho]
    confianza = float(resultado[indice_predicho]) * 100

    return clase_predicha, confianza, resultado


def main():
    parser = argparse.ArgumentParser(description="Predice la clase de semáforo de una o varias imágenes")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--imagen", help="Ruta a una sola imagen")
    grupo.add_argument("--carpeta", help="Ruta a una carpeta con varias imágenes")
    parser.add_argument("--modelo", default=MODELO_PATH, help=f"Ruta al modelo .h5 (default: {MODELO_PATH})")
    parser.add_argument("--training-dir", default=TRAINING_DIR,
                         help=f"Carpeta usada para deducir el orden de las clases (default: {TRAINING_DIR})")
    args = parser.parse_args()

    print("Cargando modelo...")
    modelo = tf.keras.models.load_model(args.modelo)

    clases = obtener_orden_clases(args.training_dir)
    print(f"Clases (en orden): {clases}\n")

    if args.imagen:
        rutas = [Path(args.imagen)]
    else:
        carpeta = Path(args.carpeta)
        rutas = sorted([p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES_VALIDAS])
        if not rutas:
            print(f"No se encontraron imágenes válidas en {carpeta}")
            return

    for ruta in rutas:
        if not ruta.exists():
            print(f"  (!) No existe: {ruta}")
            continue
        clase, confianza, probs = predecir_imagen(modelo, ruta, clases)
        detalle = ", ".join(f"{c}: {p*100:.1f}%" for c, p in zip(clases, probs))
        print(f"{ruta.name} -> {clase}  ({confianza:.1f}% confianza)")
        print(f"    [{detalle}]")


if __name__ == "__main__":
    main()
