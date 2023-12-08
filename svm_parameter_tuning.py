import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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

random_seed = 4
random.seed(random_seed)
for i in range(199):
    train_features.append(random.choice(stroke_features.iloc[249:].values.tolist()))
    train_target.append(random.choice(stroke_target.iloc[249:].values.tolist()))
for i in range(50):
    test_features.append(random.choice(stroke_features.iloc[249:].values.tolist()))
    test_target.append(random.choice(stroke_target.iloc[249:].values.tolist()))


# SVM Parameter Tuning
param_grid={'C':[0.001,0.01,1,10,100], 'gamma':[1,0.1,0.01,0.001,0.0001], 'kernel':['rbf', 'linear']}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(train_features, train_target)

print(grid.best_params_)


grid_predictions = grid.predict(test_features)

grid_accuracy = accuracy_score(test_target, grid_predictions)
grid_precision = precision_score(test_target, grid_predictions)
grid_recall = recall_score(test_target, grid_predictions)
grid_confusion_matrix = confusion_matrix(test_target, grid_predictions)

print(f'\nAccuracy: {grid_accuracy:.2f}')
print(f'Precision: {grid_precision:.2f}')
print(f'Recall: {grid_recall:.2f}')
print('Confusion Matrix:')
print(grid_confusion_matrix)