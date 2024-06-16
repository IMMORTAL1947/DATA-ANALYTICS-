import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



# Load the datasets
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')

# Merge training features and labels
train_data = pd.merge(train_features, train_labels, on='respondent_id')

# Impute missing values
# Numerical features: fill with median
numerical_features = train_data.select_dtypes(include=['float64']).columns
train_data[numerical_features] = train_data[numerical_features].fillna(train_data[numerical_features].median())

# Categorical features: fill with mode
categorical_features = train_data.select_dtypes(include=['object']).columns
train_data[categorical_features] = train_data[categorical_features].apply(lambda x: x.fillna(x.mode()[0]))

# Encode categorical variables
train_data_encoded = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Separate features and target variables
X = train_data_encoded.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'], axis=1)
y_xyz = train_data_encoded['xyz_vaccine']
y_seasonal = train_data_encoded['seasonal_vaccine']

# Split the data
X_train, X_val, y_train_xyz, y_val_xyz, y_train_seasonal, y_val_seasonal = train_test_split(
    X, y_xyz, y_seasonal, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model for xyz_vaccine
model_xyz = LogisticRegression(max_iter=1000)
model_xyz.fit(X_train, y_train_xyz)

# Predict probabilities for xyz_vaccine
y_pred_xyz = model_xyz.predict_proba(X_val)[:, 1]

# Calculate ROC AUC for xyz_vaccine
roc_auc_xyz = roc_auc_score(y_val_xyz, y_pred_xyz)

# Initialize and train the logistic regression model for seasonal_vaccine
model_seasonal = LogisticRegression(max_iter=1000)
model_seasonal.fit(X_train, y_train_seasonal)

# Predict probabilities for seasonal_vaccine
y_pred_seasonal = model_seasonal.predict_proba(X_val)[:, 1]

# Calculate ROC AUC for seasonal_vaccine
roc_auc_seasonal = roc_auc_score(y_val_seasonal, y_pred_seasonal)

# Print the ROC AUC scores
print(roc_auc_xyz, roc_auc_seasonal, (roc_auc_xyz + roc_auc_seasonal) / 2)

# Impute missing values for the test set
test_features[numerical_features] = test_features[numerical_features].fillna(test_features[numerical_features].median())
test_features[categorical_features] = test_features[categorical_features].apply(lambda x: x.fillna(x.mode()[0]))

# Encode categorical variables
test_features_encoded = pd.get_dummies(test_features, columns=categorical_features, drop_first=True)

# Ensure the test set has the same columns as the training set
missing_cols = set(train_data_encoded.columns) - set(test_features_encoded.columns)
for col in missing_cols:
    test_features_encoded[col] = 0
test_features_encoded = test_features_encoded[train_data_encoded.columns.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])]

# Predict probabilities for xyz_vaccine
test_pred_xyz = model_xyz.predict_proba(test_features_encoded)[:, 1]

# Predict probabilities for seasonal_vaccine
test_pred_seasonal = model_seasonal.predict_proba(test_features_encoded)[:, 1]

# Prepare submission
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': test_pred_xyz,
    'seasonal_vaccine': test_pred_seasonal
})

# Save submission file
submission.to_csv('submission.csv', index=False)

