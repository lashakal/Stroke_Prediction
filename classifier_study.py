import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

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


# KNN Classifier
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
# error_rate = []
# for i in range(1, 50):
#     knn = neighbors.KNeighborsClassifier(n_neighbors = i)
#     knn.fit(train_features, train_target)
#     predict_i = knn.predict(test_features)
#     error_rate.append(np.mean(predict_i != test_target))

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 50), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o')
# plt.title("Error rate VS K values of KNN")
# plt.xlabel("K values")
# plt.ylabel("Error rate")
# plt.show()

# from the graph - k=25
n_neighbors = 25
knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(train_features, train_target)
knn_predictions = knn.predict(test_features)

knn_accuracy = accuracy_score(test_target, knn_predictions)
knn_precision = precision_score(test_target, knn_predictions)
knn_recall = recall_score(test_target, knn_predictions)
knn_f1_score = f1_score(test_target, knn_predictions)

print(f'KNN Accuracy: {knn_accuracy:.2f}')
print(f'KNN Precision: {knn_precision:.2f}')
print(f'KNN Recall: {knn_recall:.2f}')
print(f'KNN F1 Score: {knn_f1_score:.2f}')


# SVM Classifier - Support Vector Machine
svm = SVC()
svm.fit(train_features, train_target)
svm_predictions = svm.predict(test_features)

svm_accuracy = accuracy_score(test_target, svm_predictions)
svm_precision = precision_score(test_target, svm_predictions)
svm_recall = recall_score(test_target, svm_predictions)
svm_f1_score = f1_score(test_target, svm_predictions)

print(f'\nSVM Accuracy: {svm_accuracy:.2f}')
print(f'SVM Precision: {svm_precision:.2f}')
print(f'SVM Recall: {svm_recall:.2f}')
print(f'SVM F1 Score: {svm_f1_score:.2f}')


# Naive Bayes
nb = GaussianNB()
nb.fit(train_features, train_target)
nb_predictions = nb.predict(test_features)

nb_accuracy = accuracy_score(test_target, nb_predictions)
nb_precision = precision_score(test_target, nb_predictions)
nb_recall = recall_score(test_target, nb_predictions)
nb_f1_score = f1_score(test_target, nb_predictions)

print(f'\nNaive Bayes Accuracy: {nb_accuracy:.2f}')
print(f'Naive Bayes Precision: {nb_precision:.2f}')
print(f'Naive Bayes Recall: {nb_recall:.2f}')
print(f'Naive Bayes F1 Score: {nb_f1_score:.2f}')


# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_features, train_target)
rf_predictions = rf.predict(test_features)

rf_accuracy = accuracy_score(test_target, rf_predictions)
rf_precision = precision_score(test_target, rf_predictions)
rf_recall = recall_score(test_target, rf_predictions)
rf_f1_score = f1_score(test_target, rf_predictions)

print(f'\nRandom Forest Accuracy: {rf_accuracy:.2f}')
print(f'Random Forest Precision: {rf_precision:.2f}')
print(f'Random Forest Recall: {rf_recall:.2f}')
print(f'Random Forest F1 Score: {rf_f1_score:.2f}')