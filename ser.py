from numpy.lib.function_base import average
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle

from utils import load_data

print("Loading dataset")
# load RAVDESS dataset

# y corresponds to emotion
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
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
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)
# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
print(f'F1 score: {f1_score(y_test, y_pred,average="macro")}')
print(f'Precision score: {precision_score(y_test, y_pred,average="macro")}')

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))


cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='')

plt.xlabel="Predicted Label"
plt.xticks(plt.xticks()[0], labels=model.classes_)

plt.ylabel="Actual label"
plt.yticks(plt.yticks()[0], labels=model.classes_, rotation=0)

plt.show()
