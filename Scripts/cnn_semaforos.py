# Importando librerías
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

print(tf.__version__)

# =====================================================================
# IMPORTANTE - Estructura de carpetas esperada:
#
#   dataset/training_set/rojo/...
#   dataset/training_set/verde/...
#   dataset/training_set/sin_semaforo/...
#
#   dataset/test_set/rojo/...
#   dataset/test_set/verde/...
#   dataset/test_set/sin_semaforo/...
#

# Parte 1 - Preprocesamiento de datos

# Preprocesamiento del training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='categorical')  # <- cambio: antes 'binary'

# Preprocesamiento del test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='categorical')  # <- cambio: antes 'binary'

num_clases = training_set.num_classes
print(f"Clases detectadas: {training_set.class_indices}")

# Parte 2 - Construcción de la CNN

cnn = tf.keras.models.Sequential()

# Paso 1 - Convolución
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Paso 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Segunda capa convolucional
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Paso 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Paso 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Un poco de Dropout ayuda a evitar overfitting con datasets pequeños
cnn.add(tf.keras.layers.Dropout(0.3))

# Paso 5 - Capa de salida
# <- cambio: antes 1 neurona + sigmoid (binario), ahora N neuronas + softmax (multi-clase)
cnn.add(tf.keras.layers.Dense(units=num_clases, activation='softmax'))

# Parte 3 - Entrenamiento de la CNN

# <- cambio: antes 'binary_crossentropy', ahora 'categorical_crossentropy'
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Guardar el modelo entrenado para usarlo después sin reentrenar
cnn.save('modelo_semaforos.h5')
print("Modelo guardado como 'modelo_semaforos.h5'")

# Parte 4 - Haciendo una predicción individual

# Mapeo inverso: {0: 'rojo', 1: 'sin_semaforo', 2: 'verde'} (el orden real
# depende del orden alfabético de tus carpetas, por eso usamos class_indices)
indice_a_clase = {v: k for k, v in training_set.class_indices.items()}

test_image = image.load_img('dataset/single_prediction/foto_prueba.jpg')
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # <- importante: reescalar igual que en el entrenamiento

result = cnn.predict(test_image)
clase_predicha = indice_a_clase[np.argmax(result[0])]
confianza = np.max(result[0]) * 100

print(f"Predicción: {clase_predicha} ({confianza:.1f}% de confianza)")
print(f"Probabilidades completas: {dict(zip(training_set.class_indices.keys(), result[0]))}")
