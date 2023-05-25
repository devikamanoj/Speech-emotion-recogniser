#  Train the model wrt dataset

from sklearn.neural_network import MLPClassifier #to make model
from sklearn.metrics import accuracy_score, f1_score, precision_score,confusion_matrix
import numpy as np
import itertools

import matplotlib.pyplot as plt
import os
import pickle

from utils import load_data

# load RAVDESS dataset
print("\n")
print("   !!! LOADING DATASET !!!")
print("\n")

# y corresponds to emotion
# x for feature
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

# print some details
print(" Number of training samples:", X_train.shape[0])
print(" Number of testing samples:", X_test.shape[0])

# this is a vector of features extracted
# using utils.extract_features() method
print(" Number of features:", X_train.shape[1])

# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 600,
}
# using best parameters so far, init MLP classifier
model = MLPClassifier(**model_params)

# train the model
print()
print("   !!! TRAINING THE MODEL !!!")
print("\n")
model.fit(X_train, y_train)

# predict 25% of data
y_pred = model.predict(X_test)

# evaluation metrics
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("\n")
print(" EVALUATION METRICES: ")
print("\n")
print("Accuracy: {:.2f}%".format(accuracy*100))
print(f'F1 score: {f1_score(y_test, y_pred,average="macro")}')
print(f'Precision score: {precision_score(y_test, y_pred,average="macro")}')

# the model is saved in a directory result. The directory is made if that doesnt exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

# for future use, the model is saved using pickle module
pickle.dump(model, open("result/mlp_classifier.model", "wb"))

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = np.unique(y_test)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Label the plot
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

