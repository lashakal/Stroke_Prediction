import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

stroke_dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")
stroke_features = stroke_dataset.drop(['id', 'stroke'], axis=1)
stroke_target = stroke_dataset['stroke']

# Preprocessing
# bmi has N/A values for some - replace with mean of the column
mean_bmi = stroke_features['bmi'].mean()
stroke_features['bmi'].fillna(mean_bmi, inplace=True)

# Label encoding - encoding string literals into integer values
label_encoder = LabelEncoder()
columns_to_encode = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for column in columns_to_encode:
    stroke_features[column] = label_encoder.fit_transform(stroke_features[column])

# Split the dataset - 80% for training and 20% for testing
# 199 for training and 50 for testing
train_features = []
train_target = []
test_features = []
test_target = []

for i in range(0, 199):
    train_features.append(stroke_features.iloc[i])
    train_target.append(stroke_target.iloc[i])
for i in range(199, 249):
    test_features.append(stroke_features.iloc[i])
    test_target.append(stroke_target.iloc[i])
for i in range(249, 448):
    train_features.append(stroke_features.iloc[i])
    train_target.append(stroke_target.iloc[i])
for i in range(448, 498):
    test_features.append(stroke_features.iloc[i])
    test_target.append(stroke_target.iloc[i])


# KNN Classifier
# finding appropriate value of k
error_rate = []
for i in range(1, 40):
    knn = neighbors.KNeighborsClassifier(n_neighbors = i)
    knn.fit(train_features, train_target)
    predict_i = knn.predict(test_features)
    error_rate.append(np.mean(predict_i != test_target))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o')
plt.title("Error rate VS K values of KNN")
plt.xlabel("K values")
plt.ylabel("Error rate")
plt.show()

# from the graph - k=25
n_neighbors = 25
knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(train_features, train_target)
knn_predictions = knn.predict(test_features)

knn_accuracy = accuracy_score(test_target, knn_predictions)
knn_precision = precision_score(test_target, knn_predictions)
knn_recall = recall_score(test_target, knn_predictions)
knn_confusion_matrix = confusion_matrix(test_target, knn_predictions)

print(f'KNN Accuracy: {knn_accuracy:.2f}')
print(f'KNN Precision: {knn_precision:.2f}')
print(f'KNN Recall: {knn_recall:.2f}')
print('KNN Confusion Matrix:')
print(knn_confusion_matrix)


# SVM Classifier - Support Vector Machine
svm = SVC(kernel='linear')
svm.fit(train_features, train_target)
svm_predictions = svm.predict(test_features)

svm_accuracy = accuracy_score(test_target, svm_predictions)
svm_precision = precision_score(test_target, svm_predictions)
svm_recall = recall_score(test_target, svm_predictions)
svm_confusion_matrix = confusion_matrix(test_target, svm_predictions)

print(f'\nSVM Accuracy: {svm_accuracy:.2f}')
print(f'SVM Precision: {svm_precision:.2f}')
print(f'SVM Recall: {svm_recall:.2f}')
print('SVM Confusion Matrix:')
print(svm_confusion_matrix)


# Naive Bayes
nb = GaussianNB()
nb.fit(train_features, train_target)
nb_predictions = nb.predict(test_features)

nb_accuracy = accuracy_score(test_target, nb_predictions)
nb_precision = precision_score(test_target, nb_predictions)
nb_recall = recall_score(test_target, nb_predictions)
nb_confusion_matrix = confusion_matrix(test_target, nb_predictions)

print(f'\nNaive Bayes Accuracy: {nb_accuracy:.2f}')
print(f'Naive Bayes Precision: {nb_precision:.2f}')
print(f'Naive Bayes Recall: {nb_recall:.2f}')
print('Naive Bayes Confusion Matrix:')
print(nb_confusion_matrix)


# Decision Tree classifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(train_features, train_target)
dt_predictions = dt.predict(test_features)

dt_accuracy = accuracy_score(test_target, dt_predictions)
dt_precision = precision_score(test_target, dt_predictions)
dt_recall = recall_score(test_target, dt_predictions)
dt_confusion_matrix = confusion_matrix(test_target, dt_predictions)

print(f'\nDecision Tree Accuracy: {dt_accuracy:.2f}')
print(f'Decision Tree Precision: {dt_precision:.2f}')
print(f'Decision Tree Recall: {dt_recall:.2f}')
print('Decision Tree Confusion Matrix:')
print(dt_confusion_matrix)