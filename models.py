"""
Definizione dei modelli e delle relative griglie di iperparametri per la grid search.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, cohen_kappa_score


# Definizione delle metriche di scoring
SCORING_METRICS = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'f1_macro': 'f1_macro',
    'cohen_kappa': make_scorer(cohen_kappa_score)
}

# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================

def get_logreg_config():
    """
    Configurazione per Logistic Regression.
    
    Returns:
        tuple: (estimator, param_grid)
    """
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2']
    }
    
    return estimator, param_grid


# ============================================================================
# SUPPORT VECTOR MACHINE
# ============================================================================

def get_svm_config():
    """
    Configurazione per Support Vector Machine.
    
    Returns:
        tuple: (estimator, param_grid)
    """
    estimator = SVC(random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    return estimator, param_grid


# ============================================================================
# RANDOM FOREST
# ============================================================================

def get_rf_config():
    """
    Configurazione per Random Forest.
    
    Returns:
        tuple: (estimator, param_grid)
    """
    estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    return estimator, param_grid


# ============================================================================
# K-NEAREST NEIGHBORS
# ============================================================================

def get_knn_config():
    """
    Configurazione per K-Nearest Neighbors.
    
    Returns:
        tuple: (estimator, param_grid)
    """
    estimator = KNeighborsClassifier(n_jobs=-1)
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'cosine']
    }
    
    return estimator, param_grid


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY = {
    'logreg': get_logreg_config,
    'svm': get_svm_config,
    'rf': get_rf_config,
    'knn': get_knn_config
}


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def get_model_and_grid(model_name, refit_metric='accuracy', cv=5, verbose=0):
    """
    Restituisce un GridSearchCV configurato per il modello specificato.
    
    Args:
        model_name (str): Nome del modello ('logreg', 'svm', 'rf', 'knn')
        refit_metric (str): Metrica da usare per selezionare il miglior modello
        cv (int): Numero di fold per la cross-validation
        verbose (int): Livello di verbosità
        
    Returns:
        GridSearchCV: Oggetto grid search configurato
        
    Raises:
        ValueError: Se il modello non è riconosciuto
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modello '{model_name}' non riconosciuto. "
            f"Modelli disponibili: {list(MODEL_REGISTRY.keys())}"
        )
    
    # Ottieni estimator e param_grid
    estimator, param_grid = MODEL_REGISTRY[model_name]()
    
    # Crea GridSearchCV
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=SCORING_METRICS,
        refit=refit_metric,
        n_jobs=-1,
        verbose=verbose
    )
    
    return grid


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_available_models():
    """
    Restituisce la lista dei modelli disponibili.
    
    Returns:
        list: Lista dei nomi dei modelli disponibili
    """
    return list(MODEL_REGISTRY.keys())


def get_param_grid(model_name):
    """
    Restituisce solo la griglia di parametri per un modello specifico.
    
    Args:
        model_name (str): Nome del modello
        
    Returns:
        dict: Griglia di parametri
        
    Raises:
        ValueError: Se il modello non è riconosciuto
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modello '{model_name}' non riconosciuto. "
            f"Modelli disponibili: {list(MODEL_REGISTRY.keys())}"
        )
    
    _, param_grid = MODEL_REGISTRY[model_name]()
    return param_grid
