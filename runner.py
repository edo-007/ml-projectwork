from utils import *

import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


DEF_SEED = 42

def main(args):

    import warnings

    # Controllo coerenza tra nrows e seed
    if args.nrows == "all":
        warnings.warn(
            Colors.YELLOW
                + "Nessun campionamento delle righe, intero dataset utilizzato"
                + Colors.END
        )
    else:
        # Qui si effettua un campionamento
        if args.seedsamp < 0:
            raise ValueError(
                Colors.RED
                    + "È richiesto un seed valido (>= 0) per il campionamento delle righe."
                    + Colors.END
            )
        
    if args.seedsplit == DEF_SEED:
        warnings.warn(
            Colors.YELLOW
                + f"Seed per lo split train/test settato automaticamente a {DEF_SEED}"
                + Colors.END
        )

    if args.data is None and args.embs is None:
        raise ValueError(
            Colors.RED + "Nessun input fornito: specificare --data oppure --embs." + Colors.END
        )

    if args.data is not None and args.embs is not None:
        warnings.warn(
            Colors.YELLOW
            + "Specificati sia --data che --embs. Verrà usato il dataset (--data) come sorgente principale."
            + Colors.END
        )

    if not 0 < args.testsize < 1:
        raise ValueError(
            Colors.RED + "Dimensione del test non consentita" + Colors.END
        )


    embs = get_embeddings(args)

   # Selezione modello
    match args.model:
        case "knn":
            clf = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': [3, 4, 5],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }

        case "logreg":
            clf = LogisticRegression(max_iter=1000)
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'saga'],
                'penalty': ['l2']
            }

        case "svm":
            clf = LinearSVC(max_iter=5000)
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'loss': ['hinge', 'squared_hinge']
            }

        case _:
            raise ValueError(f"Modello '{args.model}' non supportato. Usa: 'knn', 'logreg', o 'svm'.")


    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit="accuracy",
        n_jobs=-1,
        verbose=1
    )    

    grid.fit(embs['X_train'], embs['y_train'])

    ###########################################################################

    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(embs['X_test'])


    print("Distribuzione PREDIZIONE (y_pred):")
    print({label: count for label, count in Counter(y_pred).items()})

    print("\n" + "="*50)
    print("Migliori iperparametri:", grid.best_params_)
    print(f"Miglior score CV: {grid.best_score_:.4f}")
    print("="*50)

    # Valutazione finale
    
    print("\n" + "="*50)
    print("Classification Report (Test Set):")
    print("="*50)
    print(classification_report(embs['y_test'],  y_pred))

    results = pd.DataFrame(grid.cv_results_)
    metriche = [col for col in results.columns]
    print(results[metriche])

    # Salvataggio modello migliore

    model_filename = f"best_knn_{Path(args.data).stem}.joblib"
    joblib.dump(best_clf, model_filename)
    print(f"\nModello salvato in: {model_filename}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classificatore KNN con embedding SBERT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data", "-d", 
        default=None,
        help="Percorso del dataset CSV"
    )
    
    parser.add_argument(
        "--embs", "-e", 
        default=None,
        help="Percorso del file .joblib con embedding pre-calcolati"
    )

    parser.add_argument(
        "--nrows", "-nr", 
        default="all", 
        help="Numero di righe da prelevare dal dataset (bilanciato per classe)"
    )
    
    parser.add_argument(
        "--seedsamp", 
        type=int,
        default=-1, 
        help="Seed per riproducibilità sampling delle istanze se nrow != all"
    )
    
    parser.add_argument(
        "--seedsplit", 
        type=int,
        default=DEF_SEED, 
        help="Seed per riproducibilità per lo split train/test"
    )

    parser.add_argument(
        "--strans", 
        default="all-mpnet-base-v2",
        help="Modello SentenceTransformer da usare"
    )

    parser.add_argument(
        "--model", 
        default="knn",
        help="Modello di ML da usare"
    )

    parser.add_argument(
        "--testsize",
        type=float,
        default=0.20, 
        help="dimensione del dataset di test "
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"\nERRORE FATALE: {e}")
        raise
