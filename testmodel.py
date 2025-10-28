"""
Script principale per testare modelli di classificazione con grid search.
Supporta diversi classificatori: LogisticRegression, SVM, RandomForest, KNN.
"""

import argparse
import warnings
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, cohen_kappa_score
from collections import Counter

from utils import get_embeddings, Colors
from models import get_model_and_grid


def run_experiment(args):
    """
    Esegue l'esperimento completo: carica dati, esegue grid search, valuta risultati.
    
    Args:
        args: Argomenti da command line
    """
    # Caricamento embeddings
    embs = get_embeddings(args)
    
    # Recupera modello e grid search configurati
    grid = get_model_and_grid(args.model, args.refit)
    
    # Training con grid search
    print(Colors.BOLD + Colors.PURPLE + f"\nAvvio Grid Search per {args.model.upper()}..." + Colors.END)
    grid.fit(embs['X_train'], embs['y_train'])
    
    # Miglior modello
    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(embs['X_test'])
    
    # Stampa risultati
    print_results(grid, y_pred, embs['y_test'], args)
    
    # Salvataggio modello
    save_model(best_clf, args)
    
    return best_clf, grid


def print_results(grid, y_pred, y_test, args):
    """
    Stampa i risultati della grid search e della valutazione finale.
    
    Args:
        grid: Oggetto GridSearchCV fitted
        y_pred: Predizioni sul test set
        y_test: Label reali del test set
        args: Argomenti da command line
    """
    print("\nDistribuzione PREDIZIONE (y_pred):")
    print({label: count for label, count in Counter(y_pred).items()})
    
    print("\n" + "="*50)
    print("Migliori iperparametri:", grid.best_params_)
    print(f"Miglior score ({args.refit}) CV: {grid.best_score_:.4f}")
    print("="*50)
    
    print("\n" + "="*50)
    print("Classification Report (Test Set):")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    if args.refit == 'cohen_kappa':
        print(f"Cohen Kappa (test): {cohen_kappa_score(y_test, y_pred):.4f}")


def save_model(model, args):
    """
    Salva il modello addestrato su disco.
    
    Args:
        model: Modello da salvare
        args: Argomenti da command line
    """
    model_filename = f"models/{args.model}_data-{Path(args.data).stem}_refit-{args.refit}.joblib"
    joblib.dump(model, model_filename)
    print(f"\n{Colors.GREEN}Modello salvato in: {model_filename}{Colors.END}")


def parse_arguments():
    """
    Parsing degli argomenti da command line.
    
    Returns:
        Namespace con gli argomenti parsati
    """
    parser = argparse.ArgumentParser(
        description="Classificatore con embedding SBERT e grid search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,

        help="Percorso del dataset CSV (richiesto: colonne 'clean_text' e 'classificazione')"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="logreg",
        choices=["logreg", "svm", "rf", "knn"],
        help="Modello da utilizzare: logreg (Logistic Regression), svm (SVM), rf (Random Forest), knn (KNN)"
    )
    
    parser.add_argument(
        "--refit",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_macro", "cohen_kappa"],
        help="Metrica per selezionare il miglior modello nella grid search"
    )
    
    parser.add_argument(
        "--testsize",
        type=float,
        default=0.2,

        help="Proporzione del dataset da usare per il test set (0.0-1.0)"
    )
    
    parser.add_argument(
        "--seedsplit",
        type=int,
        default=42,
        help="Seed per la riproducibilità dello split train/test"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """
    Valida gli argomenti da command line.
    
    Args:
        args: Argomenti parsati
        
    Raises:
        ValueError: Se gli argomenti non sono validi
    """
    if args.testsize <= 0 or args.testsize >= 1:
        raise ValueError(
            f"{Colors.RED}--testsize deve essere tra 0 e 1 (esclusivi){Colors.END}"
        )
    
    if args.seedsplit < 0:
        raise ValueError(
            f"{Colors.RED}--seedsplit deve essere un intero non negativo{Colors.END}"
        )


def main():
    """
    Funzione principale.
    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    args = parse_arguments()
    
    try:
        validate_arguments(args)
        run_experiment(args)
        
    except FileNotFoundError as e:
        print(f"\n{Colors.RED}ERRORE: {e}{Colors.END}")
        
    except ValueError as e:
        print(f"\n{Colors.RED}ERRORE: {e}{Colors.END}")
        
    except Exception as e:
        print(f"\n{Colors.RED}ERRORE FATALE: {e}{Colors.END}")
        raise


if __name__ == '__main__':
    main()
