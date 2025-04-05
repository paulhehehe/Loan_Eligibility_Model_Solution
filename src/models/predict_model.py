# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# # Function to predict and evaluate
def evaluate_model(model, xtest, ytest):
    
    ypred = model.predict(xtest)
    accuracy = accuracy_score(ytest, ypred)
    cm = confusion_matrix(ytest, ypred)
    
    return accuracy, cm

def cross_validate_model(model, xtrain, ytrain, cv=5):
   
    scores = cross_val_score(model, xtrain, ytrain, cv=cv)
    return scores.mean(), scores.std()