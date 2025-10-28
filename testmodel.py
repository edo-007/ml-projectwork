from utils import *

import argparse
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, cohen_kappa_score, make_scorer


DEF_SEED = 42

def main(args):

    embs = get_embeddings(args)

    clf = LogisticRegression(max_iter=1000)
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2']
    }    


    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'cohen_kappa': make_scorer(cohen_kappa_score)
    }

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit=args.refit,  
        n_jobs=-1,
        verbose=1
    )

    grid.fit(embs['X_train'], embs['y_train'])

    ###########################################################################
    global best_clf
    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(embs['X_test'])


    print("Distribuzione PREDIZIONE (y_pred):")
    print({label: count for label, count in Counter(y_pred).items()})

    print("\n" + "="*50)
    print("Migliori iperparametri:", grid.best_params_)

    # Il grid.best_score_ viene calcolato sulla metrica specificata nel parametro refit di GridSearchCV.
    print(f"Miglior score ({args.refit}) CV: {grid.best_score_:.4f}")
    print("="*50)

    # Valutazione finale
    
    print("\n" + "="*50)
    print("Classification Report (Test Set):")
    print("="*50)

    print(classification_report(embs['y_test'],  y_pred))
    print("Cohen K (in test): " + str(cohen_kappa_score(embs['y_test'],  y_pred))) if args.refit == 'cohen_kappa' else ''

    model_filename = f"models/{args.model}_nrows-{args.nrows}_refit-accuracy.joblib"
   
    # model_filename = f"best__{Path(args.data).stem}.joblib"
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
        "--refit",
        type=str,
        default="accuracy", 
    )

    import warnings
    args = parser.parse_args()
    if args.refit not in ["accuracy", "cohen_kappa"]:
            args.refit = "accuracy"
            warnings.warn(
                Colors.YELLOW
                    + f"Metrica per il refit non definita, accuratezza utilizzata"
                    + Colors.END
            )
    if args.data:
        raise ValueError(
            Colors.RED + "Nessun input fornito: specificare --data oppure --embs." + Colors.END
        )

    try:
        main(args)
    except Exception as e:
        print(f"\nERRORE FATALE: {e}")
        raise
