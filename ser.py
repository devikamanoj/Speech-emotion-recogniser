from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, plot_confusion_matrix
import matplotlib.pyplot as plt

import os
import pickle

from utils import load_data

# load RAVDESS dataset
print("Loading dataset")

# y corresponds to emotion
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

# print some details
print("[+] Number of training samples:", X_train.shape[0])
print("[+] Number of testing samples:", X_test.shape[0])

# this is a vector of features extracted
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])

# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}
# using best parameters so far, init MLP classifier
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data
y_pred = model.predict(X_test)

# evaluation metrics
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
print(f'F1 score: {f1_score(y_test, y_pred,average="macro")}')
print(f'Precision score: {precision_score(y_test, y_pred,average="macro")}')

# the model is saved in a directory result. The directory is made if that doesnt exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))

plot_confusion_matrix(model, X_test, y_test,
                      cmap=plt.cm.Blues,
                      normalize='true')
plt.title('Confusion matrix for the classifier')
plt.show()
