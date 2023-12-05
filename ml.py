from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
import sklearn
import numpy as np
import pandas


datafull = pandas.read_csv('fin.csv')
data = datafull.drop(datafull.columns[[0]], axis=1)
target = pandas.read_csv('HAM10000_metadata.csv').sort_values('image_id').get('dx')


Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3, random_state=42)

scaler = Normalizer().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)

scaler = Normalizer().fit(Xtest)
Xtest = scaler.transform(Xtest)


# Neural network classifier
mlp = MLPClassifier(alpha=1, max_iter=10000, solver='lbfgs')
mlp.fit(Xtrain, Ytrain)
# Make predictions
y_train_pred = mlp.predict(Xtrain)
y_test_pred = mlp.predict(Xtest)
# Accuracy
mlp_train_accuracy = accuracy_score(Ytrain, y_train_pred)
mlp_test_accuracy = accuracy_score(Ytest, y_test_pred) 

print(f'- MLP Train Accuracy: {mlp_train_accuracy}')
print(f'----------------------------------')
print(f'- MLP Test Accuracy: {mlp_test_accuracy}\n')


# Multiclass support vector machine algorithm
msvm_rbf = SVC(gamma=2, C=1, max_iter=10000)
msvm_rbf.fit(Xtrain, Ytrain)
# Make predictions
y_train_pred = msvm_rbf.predict(Xtrain)
y_test_pred = msvm_rbf.predict(Xtest)
# Calculate Accuracy
msvm_train_accuracy = accuracy_score(Ytrain, y_train_pred)
msvm_test_accuracy = accuracy_score(Ytest, y_test_pred) 

print(f'- MSVM Train Accuracy: {msvm_train_accuracy}')
print(f'----------------------------------')
print(f'- MSVM Test Accuracy: {msvm_test_accuracy}\n')



estimator_list = [
    ('msvm_rbf',msvm_rbf),
    ('mlp',mlp) ]

# combining algoritms
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression(max_iter=10000, solver='lbfgs')
)

# Training
stack_model.fit(Xtrain, Ytrain)
# Make predictions
y_train_pred = stack_model.predict(Xtrain)
y_test_pred = stack_model.predict(Xtest)
# Accuracy
stack_model_train_accuracy = accuracy_score(Ytrain, y_train_pred)
stack_model_test_accuracy = accuracy_score(Ytest, y_test_pred) 

print(f'- Train Accuracy: {stack_model_train_accuracy}')
print(f'----------------------------------')
print(f'- Test Accuracy: {stack_model_test_accuracy}')
