import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

model_filename = 'svm_model.joblib'
svm_model = joblib.load(model_filename)
print(f"Loaded SVM model from {model_filename}")

X = np.load('/home/cha0s/motor-alertness/human-motor-analysis/experimentation/lstm-experimentation/classifier/extracted_features/all_features.npy').astype(float)
y = np.load('/home/cha0s/motor-alertness/human-motor-analysis/experimentation/lstm-experimentation/classifier/extracted_features/all_labels.npy')

if y.dtype.kind not in ['i', 'f']:  
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

if X.shape[1] != 2:
    print("Data is not 2-dimensional. Reducing data to 2D using PCA for visualization.")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    svm_2d = SVC(C=svm_model.C, kernel=svm_model.kernel, gamma=svm_model._gamma, probability=True)
    svm_2d.fit(X_2d, y)
    model_to_plot = svm_2d
else:
    print("Data is 2-dimensional. No PCA reduction is needed.")
    X_2d = X
    model_to_plot = svm_model

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model_to_plot.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=40, cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary (Hyperplane) Visualization')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()
