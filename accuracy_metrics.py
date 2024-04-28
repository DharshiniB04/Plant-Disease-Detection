from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 
# Load the breast cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target
 
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Standardize the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
 
# Train the SVC model
model = SVC(random_state=42)
model.fit(X_train_std, y_train)
 
# Predict the test set
y_pred = model.predict(X_test_std)
 
# Create the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
 
# Create a confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()