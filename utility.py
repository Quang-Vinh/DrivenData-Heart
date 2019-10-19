import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_validate

from collections import defaultdict


# Displays confusion matrix
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, cmap=plt.cm.Blues) -> None:    
    cm = confusion_matrix(y_true, y_pred)    
    cm = pd.DataFrame(cm, index = classes, columns = classes)
    
    ax = sns.heatmap(cm, cmap = plt.cm.Blues, annot = True)
    ax.set(xlabel = 'Predicted', ylabel = 'Actual')

    
# Prints the cross validated scores
def cross_validate_scores(clf, X: np.ndarray, Y: np.ndarray, cv: int = 3, metrics: list =  ['accuracy']) -> None:
    scores_raw = cross_validate(clf, X, Y,
                                scoring = metrics,
                                n_jobs = -1,
                                cv = cv,
                                return_train_score = True,
                                verbose = True)   
    
    print('Training scores')
    for metric in metrics:
        score = scores_raw['train_' + metric]
        print('{}: {:.4f} ({:.4f})'.format(metric, score.mean(), score.std()))
        
    print('\nValidation Scores')
    for metric in metrics:
        score = scores_raw['test_' + metric]
        print('{}: {:.4f} ({:.4f})'.format(metric, score.mean(), score.std()))
    

# Prints the best scores from the grid search results
def print_best_grid_search_results(grid_search: dict) -> None:
    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_
    scorers = grid_search.scorer_
    
    print('Training scores')    
    for score in scorers:
        mean_train_score = cv_results['mean_train_' + score][best_index]
        std_train_score = cv_results['std_train_' + score][best_index]        
        print('{}: {:.8f} ({:.8f})'.format(score, mean_train_score, std_train_score))
        
    print('\nValidation scores')
    for score in scorers:
        mean_test_score = cv_results['mean_test_' + score][best_index]
        std_test_score = cv_results['std_test_' + score][best_index]
        print('{}: {:.8f} ({:.8f})'.format(score, mean_test_score, std_test_score))    

        
# Returns size of the parameters grid size
def params_grid_size(params_grid: dict) -> int:
    total = 1
    for _, values in params_grid.items():
            total *= len(values)
    return total


# Returns summary data frame from the list of tuplles of grid search results
def get_grid_search_results(grid_searches: list(tuple()), metrics: list) -> pd.DataFrame:
    models = []
    mean_train_scores = defaultdict(list)
    std_train_scores = defaultdict(list)
    mean_val_scores = defaultdict(list)
    std_val_scores = defaultdict(list)
    
    for gs in grid_searches:
        models += [gs[0]]
        cv_results = gs[1].cv_results_
        best_index = gs[1].best_index_
        
        for metric in metrics:          
            mean_train_score = cv_results['mean_train_' + metric][best_index]
            std_train_score = cv_results['std_train_' + metric][best_index]
            mean_val_score = cv_results['mean_test_' + metric][best_index]
            std_val_score = cv_results['std_test_' + metric][best_index]
            
            mean_train_scores[metric] += [mean_train_score]
            std_train_scores[metric] += [std_train_score]
            mean_val_scores[metric] += [mean_val_score]
            std_val_scores[metric] += [std_val_score]
            
    grid_search_results = pd.DataFrame({'Model': models})
    
    for metric in metrics:
        grid_search_results['mean_train_' + metric] = mean_train_scores[metric]
        grid_search_results['std_train_' + metric] = std_train_scores[metric]
        grid_search_results['mean_val_' + metric] = mean_val_scores[metric]
        grid_search_results['std_val_' + metric] = std_val_scores[metric]
    
    return grid_search_results


# Print best models for each metric
def best_models(grid_search_results: pd.DataFrame) -> pd.DataFrame:
    gs = grid_search_results.copy()
    
    gs.index = gs['Model']
    gs = gs[['mean_val_accuracy', 'mean_val_recall', 'mean_val_precision', 'mean_val_f1', 'mean_val_roc_auc']]
    
    return pd.DataFrame({
        'Model': gs.idxmax(axis=0),
        'Max': gs.max(axis=0)
    })