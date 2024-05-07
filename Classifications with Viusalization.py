import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt

def main():
    # Read and split the data
    data = pd.read_csv('./A3_TrainData.tsv', delimiter='\t')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Using PCA for dimensionality reduction
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Baseline Model - 10-fold CV Logistic Regression 
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    logistic_model = LogisticRegression(max_iter=1000)
    y_prob_lr_cv = cross_val_predict(logistic_model, X_train_pca, y_train, cv=stratified_kfold, method='predict_proba')[:, 1]
    auc_lr_cv = roc_auc_score(y_train, y_prob_lr_cv)
    f1_lr_cv = f1_score(y_train, y_prob_lr_cv >= 0.5) 
    logistic_model.fit(X_train_pca, y_train)

    # RF and SVM 
    rf_params = {'n_estimators': [50, 100], 'max_depth': [10, None]}
    svm_params = {'C': [0.1, 1], 'kernel': ['rbf']}
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=stratified_kfold, scoring='roc_auc', n_jobs=-1).fit(X_train_pca, y_train)
    grid_svm = GridSearchCV(SVC(probability=True), svm_params, cv=stratified_kfold, scoring='roc_auc', n_jobs=-1).fit(X_train_pca, y_train)

    # performence metrics 
    print_metrics(logistic_model, X_test_pca, y_test, 'Logistic Regression')
    print_best_model_metrics(grid_rf, 'Random Forest', X_test_pca, y_test)
    print_best_model_metrics(grid_svm, 'SVM', X_test_pca, y_test)
    best_rf_model = grid_rf.best_estimator_
    best_svm_model = grid_svm.best_estimator_

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest - Best": best_rf_model,
        "SVM - Best": best_svm_model
    }

    plot_curves(models, X_test_pca, y_test)

    # Selecting the best model
    best_model_info = select_best_model(grid_rf, grid_svm)
    best_model = best_model_info['model']
    print(f"The best model is {best_model_info['model_name']} with parameters {best_model_info['best_params']}")

    # Running best model with test data
    y_prob_best = best_model.predict_proba(X_test_pca)[:, 1]
    auc_best = roc_auc_score(y_test, y_prob_best)
    f1_best = f1_score(y_test, best_model.predict(X_test_pca))
    print(f"Performance of the best model on test data - AUC: {auc_best:.4f}, F1 Score: {f1_best:.4f}")

    # Addig predictions to file
    test_data = pd.read_csv('./A3_TestData.tsv', delimiter='\t')
    test_data_pca = pca.transform(test_data)
    y_test_prob = best_model.predict_proba(test_data_pca)[:, 1]
    np.savetxt('A3_predictions_202034641.txt', y_test_prob, fmt='%0.4f')

    cv_table = compile_cv_results(logistic_model, grid_rf, grid_svm, X_train_pca, y_train)
    print(cv_table)

# Functions to print metrics and plot curves
def print_metrics(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, model.predict(X_test))
    print(f"Metrics for {model_name}: AUC: {auc_score:.4f}, F1 Score: {f1:.4f}")

def print_best_model_metrics(grid, model_name, X_test, y_test):
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    print(f"Best parameters for {model_name}: {best_params}")
    print_metrics(best_model, X_test, y_test, f"{model_name} - Best")

def plot_curves(models, X_test, y_test):
    plt.figure(figsize=(12, 6))
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    for model_name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        ap = average_precision_score(y_test, y_probs)
        plt.plot(recall, precision, label=f'{model_name} - AP: {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # ROC Curve
    plt.subplot(1, 2, 2)
    for model_name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} - AUC: {roc_auc:.4f}')
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.show()

def select_best_model(grid_rf, grid_svm):
    best_model_info = {}
    if grid_rf.best_score_ > grid_svm.best_score_:
        best_model_info['model'] = grid_rf.best_estimator_
        best_model_info['model_name'] = 'Random Forest'
        best_model_info['best_params'] = grid_rf.best_params_
    else:
        best_model_info['model'] = grid_svm.best_estimator_
        best_model_info['model_name'] = 'SVM'
        best_model_info['best_params'] = grid_svm.best_params_

    return best_model_info

def compile_cv_results(logistic_model, grid_rf, grid_svm, X_train_pca, y_train):
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # CV for Logistic Regression
    cv_results_lr = cross_val_score(logistic_model, X_train_pca, y_train, cv=stratified_kfold, scoring='roc_auc')

    # AUC and std for RF and SVM
    rf_mean_auc = grid_rf.cv_results_['mean_test_score'][grid_rf.best_index_]
    rf_std_auc = grid_rf.cv_results_['std_test_score'][grid_rf.best_index_]

    svm_mean_auc = grid_svm.cv_results_['mean_test_score'][grid_svm.best_index_]
    svm_std_auc = grid_svm.cv_results_['std_test_score'][grid_svm.best_index_]

    cv_results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest - Best", "SVM - Best"],
        "Mean AUC (CV)": [cv_results_lr.mean(), rf_mean_auc, svm_mean_auc],
        "Std AUC (CV)": [cv_results_lr.std(), rf_std_auc, svm_std_auc]
    })

    return cv_results

if __name__ == "__main__":
    main()
