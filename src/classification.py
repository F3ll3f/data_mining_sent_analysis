from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,accuracy_score,make_scorer
from sklearn.model_selection import cross_val_score
from tabulate import tabulate

def print_eff_results(list_clfs_res):
    table=[["Bag-of-Words","Tf-idf","Word Embeddings"],list_clfs_res[0],list_clfs_res[1],list_clfs_res[2]]
    print(tabulate(table,headers="firstrow", tablefmt="fancy_grid",showindex=["Svm","Random-Forest","Knn"]))


true_values=[] #Keep here true values
pred_values=[] #Keep here predicted values

# Use this scorer to collect all the true values and predictions from the cross validation
def scoring_sum(y_true,y_pred):
    global true_values
    global pred_values
    true_values+=y_true.tolist()
    pred_values+=y_pred.tolist()
    return accuracy_score(y_true,y_pred)

#10-fold cross validation and evaluation
def cross_val_and_eval(model,xtrain,train_labels):
    global true_values
    global pred_values
    true_values=[]
    pred_values=[]
    kf = KFold(n_splits=10, shuffle=True)
    cross_val_score(model,xtrain,train_labels,cv=kf,scoring=make_scorer(scoring_sum))
    print(classification_report(true_values,pred_values))

def train_svm_clf(xtrain,trainlabels):
    clf_svm = svm.SVC(gamma=0.003, C=90.)
    clf_svm.fit(xtrain,trainlabels)
    return clf_svm

def test_svm_clf(xtest,model):
    return model.predict(xtest)
    
def train_random_forest_clf(xtrain,trainlabels):
    random_forest_clf = RandomForestClassifier(n_estimators=120)
    random_forest_clf.fit(xtrain,trainlabels)
    return random_forest_clf

def test_random_forest_clf(xtest,model):
    return model.predict(xtest)
    
def train_knn_clf(xtrain,trainlabels):
    knn_clf = KNeighborsClassifier(n_neighbors=6)
    knn_clf.fit(xtrain,trainlabels)
    return knn_clf

def test_knn_clf(xtest,model):
    return model.predict(xtest)

