import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
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
for i in range(249, 4137):
    train_features.append(stroke_features.iloc[i])
    train_target.append(stroke_target.iloc[i])
for i in range(4137, 5110):
    test_features.append(stroke_features.iloc[i])
    test_target.append(stroke_target.iloc[i])

# SVM
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