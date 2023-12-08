import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder

# In user input list, we need to have the following:
# gender - 0 (Female), 1 (Male)
# age
# hypertension - 0, 1
# heart_disease - 0, 1
# ever_married - 0 (No), 1 (Yes)
# work_type - 0 (Govt_job), 1 (Never_worked), 2 (Private), 3 (Self_employed), 4 (children)
# Residence_type - 0 (Rural), 1 (Urban)
# avg_glucose_level
# bmi
# smoking_status - 0 (unknown), 1 (formerly smoked), 2 (never smoked), 3 (smokes)

# for demonstration - this is what user_input list could look like
# user_input = [1, 22, 0, 0, 0, 2, 1, 90, 30, 2]


# KNN classifier
def KNN(user_input):
    stroke_features, stroke_target = preprocessing()

    train_features = []
    train_target = []

    for i in range(0, 498):
        train_features.append(stroke_features.iloc[i])
        train_target.append(stroke_target.iloc[i])

    # Train the KNN model
    knn = neighbors.KNeighborsClassifier(n_neighbors = 35)
    knn.fit(train_features, train_target)
    prediction = knn.predict([user_input])

    return prediction[0]

def preprocessing():
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

    return stroke_features, stroke_target