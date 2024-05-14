from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


models_types = {

    "RandomForest" : {
        "model": RandomForestClassifier(), 
        "params": {
            'n_estimators': [10, 50, 100]
        }
    }, 

    "SVC": {
        "model" : SVC(), 
        "params" : {
            'C': [1, 10, 100], 
            'kernel': ['linear', 'rbf']
        }    
    },

    "kNN": {
        
        "model" : KNeighborsClassifier(), 
        "params" : {
            "n_neighbors" : [3, 5, 10]
        }
    }
}


def grid_search(model:dict, params:dict, cv = 1):
    grid = GridSearchCV(model = model["model"], params = params, cv = cv)
    return grid


for model in models_types.keys():
    ...





