"""
Script per confrontare modelli al crescere delle istanze in training.
Identifica quale modello performa meglio con il minor numero di istanze.


# Esempio base - tutti i modelli
python learning_curve.py --data dataset.csv

# Solo alcuni modelli
python learning_curve.py --data dataset.csv --models logreg svm

# Dimensioni custom (percentuali)
python learning_curve.py --data dataset.csv --train-sizes 0.05 0.1 0.2 0.5 1.0

# Dimensioni assolute
python learning_curve.py --data dataset.csv --train-sizes 100 500 1000 5000

# Specificare directory output
python learning_curve.py --data dataset.csv --output-dir esperimenti/exp1

# Senza grafici (solo dati)
python learning_curve.py --data dataset.csv --no-plot

"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from collections import defaultdict
import json
from datetime import datetime

from utils import get_embeddings, Colors, load_dataset
from models import get_model_and_grid, get_parametrized_estimator, MODEL_REGISTRY

# Configurazione stile grafici
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

MIN_INSTANCES = 20

def train_and_evaluate(
        X_train_subset, 
        y_train_subset, 
        X_test, 
        y_test, 
        model_name, 
        refit='accuracy'
        ):
    """
    Addestra un modello su un subset dei dati e valuta le performance.
    
    Args:
        X_train_subset: Subset del training set
        y_train_subset: Label del subset
        X_test: Test set completoritrovo 
        y_test: Label del test set
        model_name: Nome del modello
        refit: Metrica per la grid search
        
    Returns:
        dict: Metriche di performance
    """
    try:

        # # Ottieni grid search con CV ridotta per velocità
            # grid = get_model_and_grid(model_name, refit_metric=refit, cv=3, verbose=0)
            # grid.fit(X_train_subset, y_train_subset)
            # y_pred = grid.best_estimator_.predict(X_test)
        
        estimator = get_parametrized_estimator(model_name)
        estimator.fit(X_train_subset, y_train_subset)
        y_pred = estimator.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'cohen_kappa': cohen_kappa_score(y_test, y_pred),
            # 'best_params': grid.best_params_
        }
        
        return metrics
        
    except Exception as e:
        print(f"{Colors.RED}Errore con {model_name} su {len(X_train_subset)} istanze: {e}{Colors.END}")
        return None



def _subset_indices(indices, train_size):
    np.random.shuffle(indices)
    return indices[:train_size]


def compute_learning_curves(X_train, y_train, X_test, y_test, 
            models_to_test, relative_train_sizes, refit='accuracy'):
    """
    Calcola le learning curves per tutti i modelli specificati.
    
    Args:
        X_train: Feature del training set
        y_train: Label del training set
        X_test: Feature del test set
        y_test: Label del test set
        models_to_test: Lista di modelli da testare
        train_sizes: Liste/percentuali di dimensioni del training set
        refit: Metrica per la grid search
        
    Returns:
        dict: Risultati per ogni modello e dimensione
    """

    # Creo un default dict a 2 livelli: 
    # Lv1 - Quando accedo ad una key non esistente crea un defultdict(list) (innestato) per quella key
    # Lv2 - Quando accedi ad una key che non esiste crea una lista vuota per quella key
    results = defaultdict(lambda: defaultdict(list))
    
    total_train_size = len(X_train)
    print(f"\n{Colors.BOLD}Totale campioni training: {total_train_size}{Colors.END}")
    print(f"{Colors.BOLD}Totale campioni test: {len(X_test)}{Colors.END}\n")
    
    print(f"{Colors.CYAN}Dimensioni training set da testare: {relative_train_sizes}{Colors.END}\n")
    print("="*80)
    

    # TODO: Aggiungere controllo sull percentuali passate (comprese in [0,1])
    _train_sizes = [
        {
            "rel": frac,
            "abs": int(total_train_size * frac)
        }
        for i, frac in enumerate(sorted(set(relative_train_sizes)))
    ]
    total_iterations = len(models_to_test) * len(_train_sizes)

    
    # Itero su tutti i modelli passati #########################################################################
    ##############################################################################################################
    for model_name in models_to_test:
        print(f"\n{Colors.BOLD}{Colors.BLUE}Modello: {model_name.upper()}{Colors.END}")
        print("-"*80)
        
        # Itero su tutti le dimensioni dei TrainSet ################################################################
        ##############################################################################################################
        for current_i, size in enumerate(_train_sizes):

            if size['abs'] < MIN_INSTANCES:
                continue

            indices = np.arange(len(X_train))
            subset_indices = _subset_indices(indices, size['abs'])

            X_train_subset = X_train[subset_indices]
            y_train_subset = y_train[subset_indices]
            
            print(f"[{current_i}/{total_iterations}] Training con {size['rel']*100:>5}% istanze... ", end='', flush=True)
            
            # Training e valutazione
            metrics = train_and_evaluate(
                X_train_subset, y_train_subset, 
                X_test, y_test,
                model_name, refit
            )
            
            if metrics:
                results[model_name]['train_sizes'].append(size['rel'])
                results[model_name]['accuracy'].append(metrics['accuracy'])
                results[model_name]['f1_macro'].append(metrics['f1_macro'])
                results[model_name]['cohen_kappa'].append(metrics['cohen_kappa'])
                # results[model_name]['best_params'].append(metrics['best_params'])
                
                print(f"{Colors.GREEN}✓ Accuracy: {metrics['accuracy']:.4f}{Colors.END}")
            else:
                print(f"{Colors.RED}✗ Fallito{Colors.END}")
    
    print("\n" + "="*80)
    return dict(results)


def plot_learning_curves(results, relative_train_sizes, output_dir='results'):
    """
    Crea grafici delle learning curves per tutti i modelli.
    
    Args:
        results: Dizionario con i risultati
        output_dir: Directory dove salvare i grafici
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    metrics = ['accuracy', 'f1_macro', 'cohen_kappa']
    metric_labels = {
        'accuracy': 'Accuracy',
        'f1_macro': 'F1 Score (Macro)',
        'cohen_kappa': "Cohen's Kappa"
    }
    
    _, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot per ogni metrica
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for model_name, model_results in results.items():
            train_sizes = model_results['train_sizes']
            scores = model_results[metric]
            
            sizes = relative_train_sizes
            if len(relative_train_sizes) != len(train_sizes):
                sizes = train_sizes

            ax.plot(sizes, scores, marker='o', linewidth=2, 
                   label=model_name.upper(), markersize=8)
        
        ax.set_xlabel('% TrainingSet', fontsize=12, fontweight='bold')
        # ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_title(f'Learning Curve - {metric_labels[metric]}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Tabella riassuntiva nel quarto subplot
    ax = axes[3]
    ax.axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/learning_curves_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{Colors.GREEN}Grafico salvato in: {output_file}{Colors.END}")
    
    plt.close()


def create_comparison_table(results, output_dir='results'):
    """
    Crea una tabella comparativa dei risultati.
    
    Args:
        results: Dizionario con i risultati
        output_dir: Directory dove salvare la tabella
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Crea DataFrame per ogni metrica
    comparison_data = []
    
    for model_name, model_results in results.items():
        for i, train_size in enumerate(model_results['train_sizes']):
            comparison_data.append({
                'Model': model_name.upper(),
                'Train Size': train_size,
                'Accuracy': model_results['accuracy'][i],
                'F1 Macro': model_results['f1_macro'][i],
                'Cohen Kappa': model_results['cohen_kappa'][i]
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Salva CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{output_dir}/comparison_table_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"{Colors.GREEN}Tabella salvata in: {csv_file}{Colors.END}")
    
    # Stampa tabella formattata
    print(f"\n{Colors.BOLD}TABELLA COMPARATIVA{Colors.END}")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df


# def find_best_model_per_size(results):
#     """
#     Identifica il miglior modello per ogni dimensione del training set.
#     Se hai testato più modelli su diversi set di addestramento (con dimensioni diverse), 
#     questa funzione ti dice quale modello è il migliore per ogni dimensione, separatamente per ciascuna metrica.
#
#     Args:
#         results: Dizionario con i risultati
#     """
#     print(f"\n{Colors.BOLD}{Colors.CYAN}ANALISI: Miglior modello per ogni dimensione{Colors.END}")
#     print("="*100)
#
#     # Raggruppa per dimensione
#     all_sizes = set()
#     for model_results in results.values():
#         all_sizes.update(model_results['train_sizes'])
#
#     for size in sorted(all_sizes):
#         print(f"\n{Colors.BOLD}Training Size: {size} istanze{Colors.END}")
#
#         for metric in ['accuracy', 'f1_macro', 'cohen_kappa']:
#             best_model = None
#             best_score = -1
#
#             for model_name, model_results in results.items():
#                 if size in model_results['train_sizes']:
#                     idx = model_results['train_sizes'].index(size)
#                     score = model_results[metric][idx]
#
#                     if score > best_score:
#                         best_score = score
#                         best_model = model_name
#
#             print(f"  {metric:>15}: {best_model.upper():<10} ({best_score:.4f})")


def save_results_json(results, output_dir='results'):
    """
    Salva i risultati in formato JSON.
    
    Args:
        results: Dizionario con i risultati
        output_dir: Directory dove salvare il JSON
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"{output_dir}/results_{timestamp}.json"
    
    # Converti in formato serializzabile
    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            'train_sizes': model_results['train_sizes'],
            'accuracy': model_results['accuracy'],
            'f1_macro': model_results['f1_macro'],
            'cohen_kappa': model_results['cohen_kappa'],
            # 'best_params': [str(params) for params in model_results['best_params']]
        }
    
    with open(json_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"{Colors.GREEN}Risultati JSON salvati in: {json_file}{Colors.END}")


def parse_arguments():
    """
    Parsing degli argomenti da command line.
    """
    parser = argparse.ArgumentParser(
        description="Confronto modelli al crescere delle istanze in training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Percorso del dataset CSV"
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs='+',
        # default=['logreg', 'svm', 'rf', 'knn'],
        default=['logreg', 'svm'],
        choices=list(MODEL_REGISTRY.keys()),
        help="Modelli da confrontare (spazio-separati)"
    )
    
    parser.add_argument(
        "--train-sizes", "-t",
        nargs='+',
        type=float,
        default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        # default=[0.5, 0.7],
        help="Dimensioni del training set (percentuali 0-1 o numeri assoluti)"
    )
    
    parser.add_argument(
        "--refit",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_macro", "cohen_kappa"],
        help="Metrica per la grid search"
    )
    
    parser.add_argument(
        "--testsize",
        type=float,
        default=0.2,
        help="Proporzione del dataset per il test set"
    )
    
    parser.add_argument(
        "--seedsplit",
        type=int,
        default=42,
        help="Seed per la riproducibilità"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Directory dove salvare i risultati"
    )
    
    parser.add_argument(
        "--no-plot",
        action='store_true',
        help="Non generare grafici"
    )
    
    return parser.parse_args()


def main():
    """
    Funzione principale.
    """
    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    
    try:
        # Imposta seed per riproducibilità
        np.random.seed(args.seedsplit)
        
        print(f"\n{Colors.BOLD}{Colors.PURPLE}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.PURPLE}CONFRONTO MODELLI - LEARNING CURVES{Colors.END}")
        print(f"{Colors.BOLD}{Colors.PURPLE}{'='*80}{Colors.END}")
        
        # Carica embeddings
        embs = get_embeddings(args)

        # Calcola learning curves
        results = compute_learning_curves(

            embs['X_train'], embs['y_train'],
            embs['X_test'],  embs['y_test'],
            args.models,
            args.train_sizes,
            args.refit
        )
        
        # Crea directory output
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Analisi risultati
        # find_best_model_per_size(results)
        # Crea tabella comparativa
        # df_comparison = create_comparison_table(results, args.output_dir)
        # Salva risultati JSON
        # save_results_json(results, args.output_dir)
        
        # Genera grafici
        if not args.no_plot:
            plot_learning_curves(results, args.train_sizes, args.output_dir) 
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}Analisi completata con successo!{Colors.END}")
        print(f"{Colors.GREEN}Risultati salvati in: {args.output_dir}/{Colors.END}\n")
        
    except Exception as e:
        print(f"\n{Colors.RED}ERRORE FATALE: {e}{Colors.END}")
        raise


if __name__ == '__main__':
    main()





