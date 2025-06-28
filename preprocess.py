import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_dir = "dataset"
X, y, class_names = [], [], sorted(os.listdir(data_dir))

for idx, gesture in enumerate(class_names):
    folder_path = os.path.join(data_dir, gesture)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {img_path} (could not read)")
            continue

        img = cv2.resize(img, (64, 64))           # Resize
        img = img / 255.0                         # Normalize
        X.append(img)
        y.append(idx)

X = np.array(X)
y = to_categorical(y, num_classes=len(class_names))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

np.savez("data.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, class_names=class_names)
print("✅ Dataset preprocessing complete.")
