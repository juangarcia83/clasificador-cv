Aquí tienes el `README.md` completo y listo para **copiar, pegar y subir a tu repositorio de GitHub**:

---

```markdown
# 🖼️ Clasificador de Imágenes con Bag of Visual Words y Voting Classifier

Este proyecto implementa un sistema de clasificación de imágenes basado en el modelo de **Bag of Visual Words (BoVW)**, utilizando **clustering con K-Means**, extracción de **patches de imagen**, y un **clasificador por votación** con **SVM** y **MLP**. Fue desarrollado como ejercicio práctico para clasificar la posición de manos (izquierda/derecha, frontal/trasera) a partir de imágenes RGB.

---

## 🚀 Requisitos

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- scikit-learn

Instalación rápida:

```bash
pip install numpy opencv-python scikit-learn
````

---

## 🧠 Descripción del Funcionamiento

### 1. 📥 Carga de datos

El script toma como entrada dos carpetas:

* `TRAIN_DIR`: imágenes etiquetadas con el nombre codificando la clase (`XXXX_LF.jpg`, `XXXX_RF.jpg`, etc.)
* `TEST_DIR`: imágenes sin etiqueta

### 2. 🧩 Extracción de Patches

Cada imagen se divide en **50 patches aleatorios** de tamaño 5x5 utilizando `image.extract_patches_2d` de scikit-learn.

---

### 3. 🔄 Normalización y Clustering (BoVW)

* Todos los patches se **normalizan** con `StandardScaler`.
* Se entrena un modelo `MiniBatchKMeans` con `n_clusters = 81`.
* Cada imagen se representa como un vector de frecuencias de aparición de los clusters: su **representación Bag of Visual Words**.

---

### 4. 🧪 Clasificación

Se utiliza un **VotingClassifier** con:

* `SVC` (Support Vector Classifier)
* `MLPClassifier` (Perceptrón multicapa)

El clasificador se entrena con las representaciones obtenidas por BoVW y predice la clase de cada imagen de test.

---

### 5. 🔍 Predicción

Para las imágenes de test:

* Se extraen patches y se transforman en vectores BoVW.
* Se hacen predicciones con el clasificador entrenado.
* Los resultados se escriben en el archivo `resultados.txt` con el formato:

```
0001 LF_0
0002 RB_1
...
```

Donde `0001` es el identificador de la imagen y `LF_0` es la clase predicha con índice incremental.

---

## 📌 Etiquetas utilizadas

Las clases están codificadas en el nombre del archivo (posiciones 6-7):

| Código | Clase       |
| ------ | ----------- |
| `LB`   | Left Back   |
| `LF`   | Left Front  |
| `RB`   | Right Back  |
| `RF`   | Right Front |

---

## 🛠️ Ejecución

Desde consola, ejecutar el script con las rutas de entrenamiento y test:

```bash
python3 E5.py train/ test/ > resultados.txt
```

Asegúrate de que las carpetas `train/` y `test/` existen y contienen imágenes `.jpg`.


## 📄 Autor

**Juan García**
Estudiante del Grado en Ciencia de Datos
Universitat Politècnica de València
📧 [jgrrea@upv.es](mailto:jgrrea@upv.es)

---
```
