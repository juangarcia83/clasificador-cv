import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

# VERIFICAR ARGUMENTOS
if len(sys.argv) != 3:
    print("Uso: python3 E5.py TRAIN_DIR TEST_DIR > resultados.txt")
    sys.exit(1)

train_path = sys.argv[1]
test_path = sys.argv[2]

if not os.path.isdir(ruta_train) or not os.path.isdir(ruta_test):
    print("Error: Las rutas TRAIN_DIR o TEST_DIR no son vÃ¡lidas.")
    sys.exit(1)

# NÃºmero de patches y tamaÃ±o de cada patch
NUM_PATCHES = 50
PATCH_SIZE = (5, 5)

# NÃºmero de clusters
N_CLUSTERS = 81

label_to_num = {'LB': 0, 'LF': 1, 'RB': 2, 'RF': 3}
num_to_label = {v: k for k, v in label_to_num.items()}

# 1) CARGA DE DATOS DE ENTRENAMIENT

train_jpg_files = [f for f in os.listdir(train_path) if f.lower().endswith('.jpg')]
train_patches_list = []  # Para guardar los patches de cada imagen
train_labels = []        # Para guardar la etiqueta de cada imagen
train_filenames = []     # Nombres de archivo (opcional, aquÃ­ no se usan)

for filename in train_jpg_files:
    img_path = os.path.join(train_path, filename)
    
    with open(img_path, 'rb') as infile:
        img = cv2.imread(img_path)
        if img is None:
            continue 

    # Convertir a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    these_patches = image.extract_patches_2d(img_rgb, PATCH_SIZE,
                                             max_patches=NUM_PATCHES,
                                             random_state=0)
    these_patches = np.reshape(these_patches, (len(these_patches), -1))
    
    # AÃ±adir esta lista de patches a la lista general
    train_patches_list.append(these_patches)

    # Extraer la etiqueta del nombre (posiciones [6:8])
    img_label = filename[6:8]
    if img_label not in label_to_num:
        continue
    train_labels.append(label_to_num[img_label])

    train_filenames.append(filename)


train_patches_array = np.concatenate(train_patches_list, axis=0)  


# 2) NORMALIZACIÃ“N (StandardScaler)
scaler = StandardScaler()
train_patches_array_norm = scaler.fit_transform(train_patches_array)

# 3) ENTRENAR KMEANS para obtener BAG OF VISUAL WORDS

kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=0, verbose=False)
kmeans.partial_fit(train_patches_array_norm)

# 4) OBTENER LA REPRESENTACIÃ“N DE CADA IMAGEN DE ENTRENAMIENTO

X_train = []
for patches_img in train_patches_list:
    # Normalizamos con el mismo scaler entrenado
    patches_norm = scaler.transform(patches_img)
    # Obtenemos el Ã­ndice de cluster para cada patch
    cluster_idx = kmeans.predict(patches_norm)  # (NUM_PATCHES,)
    # Guardamos este vector como representaciÃ³n de la imagen
    X_train.append(cluster_idx)

# Convirtiendo a array numpy para entrenar el clasificador
X_train = np.array(X_train)       
y_train = np.array(train_labels)  

# 5) DEFINIR Y ENTRENAR EL VOTING CLASSIFIER (SVM + MLP)

svm_clf = SVC(probability=True, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('svm', svm_clf), ('mlp', mlp_clf)],
    voting='soft'  # 'soft' para promediar probabilidades;
)

voting_clf.fit(X_train, y_train)


# 6) PROCESAR DATOS DE TEST 

test_jpg_files = [f for f in os.listdir(test_path) if f.lower().endswith('.jpg')]
test_patches_list = []  # Parches de cada imagen de test
test_filenames = []     # Para guardar nombres

for filename in test_jpg_files:
    img_path = os.path.join(test_path, filename)
    
    with open(img_path, 'rb') as infile:
        img = cv2.imread(img_path)
        if img is None:
            continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    these_patches = image.extract_patches_2d(img_rgb, PATCH_SIZE,
                                             max_patches=NUM_PATCHES,
                                             random_state=0)
    these_patches = np.reshape(these_patches, (len(these_patches), -1))

    test_patches_list.append(these_patches)
    test_filenames.append(filename)


X_test = []
for patches_img in test_patches_list:
    patches_norm = scaler.transform(patches_img)
    cluster_idx = kmeans.predict(patches_norm)
    X_test.append(cluster_idx)

X_test = np.array(X_test)  # (n_imagenes_test, NUM_PATCHES)


# 7) HACER PREDICCIONES Y GUARDAR EN ARCHIVO .TXT
test_predictions = voting_clf.predict(X_test) 

output_file = 'resultados.txt'
count = 0  # Iniciamos un contador para las imÃ¡genes
with open(output_file, 'w', encoding='utf-8') as f:
    for filename, pred_label_num in zip(test_filenames, test_predictions):
        # Extraer los primeros 4 dÃ­gitos del nombre de archivo como ID
        file_id = filename[0:4]
        pred_label_str = num_to_label[pred_label_num]
        line = f"{file_id} {pred_label_str}_{count}\n"
        f.write(line)
        count += 1

print(f"ClasificaciÃ³n de test finalizada. Resultados guardados en {output_file}")