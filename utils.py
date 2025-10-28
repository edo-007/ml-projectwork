"""
Funzioni di utilità per il caricamento dati, generazione embeddings e validazione.
"""

import os
from pathlib import Path
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from collections import Counter


# ============================================================================
# COSTANTI
# ============================================================================

EMBEDDINGS_DIR = 'embeddings'
DEFAULT_EMBEDDING_MODEL = 'all-mpnet-base-v2'


class Colors:
    """Colori per output a terminale."""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# ============================================================================
# GESTIONE EMBEDDINGS
# ============================================================================

def compute_embeddings(train_texts, test_texts, y_train, y_test, 
                      embeddings_file, model_name=DEFAULT_EMBEDDING_MODEL):
    """
    Genera gli embedding per train e test set e li salva insieme alle label.
    
    Args:
        train_texts (list): Lista di testi per il training
        test_texts (list): Lista di testi per il test
        y_train (Series): Label del training set
        y_test (Series): Label del test set
        embeddings_file (str): Percorso del file dove salvare gli embedding
        model_name (str): Nome del modello SentenceTransformer da usare
    
    Returns:
        dict: Dictionary contenente embedding e label, None in caso di errore
    """
    print(f"\n{Colors.CYAN}Download del modello SBERT: {model_name}...{Colors.END}")
    model = SentenceTransformer(model_name)

    print(f"{Colors.CYAN}Generazione degli embedding...{Colors.END}")
    try:
        train_embeddings = model.encode(train_texts, show_progress_bar=True)
        test_embeddings = model.encode(test_texts, show_progress_bar=True)
        
        embeddings = {
            'X_train': train_embeddings,
            'X_test': test_embeddings,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'metadata': {
                'model': model_name,
                'train_shape': train_embeddings.shape,
                'test_shape': test_embeddings.shape,
                'class_distribution_train': pd.Series(y_train).value_counts().to_dict(),
                'class_distribution_test': pd.Series(y_test).value_counts().to_dict()
            }
        }
        
        # Salvataggio
        os.makedirs(Path(embeddings_file).parent, exist_ok=True)
        joblib.dump(embeddings, embeddings_file)
        
        print(f"{Colors.GREEN}Embedding salvati in: {embeddings_file}{Colors.END}")
        print(f"Shape train: {embeddings['X_train'].shape}, Shape test: {embeddings['X_test'].shape}")
        
        return embeddings
        
    except Exception as e:
        print(f"{Colors.RED}Errore durante la generazione degli embedding: {e}{Colors.END}")
        return None


def load_embeddings(embeddings_file):
    """
    Carica gli embeddings da file.
    
    Args:
        embeddings_file (str): Percorso del file degli embeddings
        
    Returns:
        dict: Dictionary contenente gli embeddings, None se il caricamento fallisce
    """
    print(f"{Colors.BOLD}{Colors.GREEN}Caricamento embedding da: {embeddings_file}{Colors.END}")
    
    try:
        embeddings = joblib.load(embeddings_file)
        
        # Verifica integrità
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test']
        if all(key in embeddings for key in required_keys):
            print(f"{Colors.GREEN}Embedding caricati correttamente!{Colors.END}")
            
            if 'metadata' in embeddings:
                print("\nMetadati embedding:")
                for key, value in embeddings['metadata'].items():
                    print(f"  {key}: {value}")
            
            return embeddings
        else:
            print(f"{Colors.RED}Embedding incompleti: mancano alcune chiavi richieste{Colors.END}")
            return None
            
    except FileNotFoundError:
        print(f"{Colors.YELLOW}File embeddings non trovato: {embeddings_file}{Colors.END}")
        return None
        
    except Exception as e:
        print(f"{Colors.RED}Errore nel caricamento degli embeddings: {e}{Colors.END}")
        return None


# ============================================================================
# GESTIONE DATASET
# ============================================================================

def validate_dataset(df):
    """
    Valida che il dataset contenga le colonne necessarie.
    
    Args:
        df (DataFrame): DataFrame da validare
    
    Raises:
        ValueError: Se mancano colonne richieste
    """
    required_cols = ['clean_text', 'classificazione']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")
    
    # Stampa informazioni sul dataset
    print(f"\n{Colors.BOLD}Informazioni dataset:{Colors.END}")
    print(f"  Righe: {len(df)}")
    print(f"  Colonne: {len(df.columns)}")
    
    print(f"\n{Colors.BOLD}Distribuzione classi:{Colors.END}")
    conteggi = df['classificazione'].value_counts()
    for chiave, valore in conteggi.items():
        print(f"  {chiave:<30} | {valore:>5} righe")
    
    print()


def load_dataset(data_path):
    """
    Carica e valida il dataset.
    
    Args:
        data_path (str): Percorso del file CSV
        
    Returns:
        DataFrame: Dataset caricato e validato
        
    Raises:
        FileNotFoundError: Se il file non esiste
        RuntimeError: Se il caricamento fallisce
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset non trovato: {data_path}")
    
    print(f"{Colors.BOLD}Caricamento dataset da: {data_path}{Colors.END}")
    
    try:
        df = pd.read_csv(data_path)
        validate_dataset(df)
        return df
        
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento del dataset: {e}")


# ============================================================================
# PIPELINE COMPLETA
# ============================================================================

def get_embeddings(args):
    """
    Funzione principale per ottenere gli embeddings.
    Carica da file se esistono, altrimenti li genera dal dataset.
    
    Args:
        args: Argomenti da command line (deve contenere: data, testsize, seedsplit)
        
    Returns:
        dict: Dictionary contenente gli embeddings
        
    Raises:
        RuntimeError: Se la generazione degli embeddings fallisce
    """
    # Controlla se gli embeddings esistono già
    embeddings_file = f"{EMBEDDINGS_DIR}/{Path(args.data).stem}.joblib"
    
    if os.path.exists(embeddings_file):
        embeddings = load_embeddings(embeddings_file)
        if embeddings is not None:
            return embeddings
        print(f"{Colors.YELLOW}Embeddings non validi, rigenerazione in corso...{Colors.END}\n")
    
    # Carica il dataset
    df = load_dataset(args.data)
    
    # Preparazione dati
    X = df['clean_text'].astype(str).tolist()
    y = df['classificazione']
    
    # input()

    # Split train/test stratificato
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.testsize, 
        random_state=args.seedsplit, 
        stratify=y
    )
    
    print(f"{Colors.BOLD}Split train/test:{Colors.END}")
    print(f"  Train: {len(X_train_txt):>6} campioni")
    print(f"  Test:  {len(X_test_txt):>6} campioni ({args.testsize*100:.1f}%)")
    
    # Distribuzione classi
    print(f"\n{Colors.BOLD}Distribuzione classi nel TRAIN:{Colors.END}")
    for label, count in Counter(y_train).items():
        print(f"  {label:<30} | {count:>5}")
    
    print(f"\n{Colors.BOLD}Distribuzione classi nel TEST:{Colors.END}")
    for label, count in Counter(y_test).items():
        print(f"  {label:<30} | {count:>5}")
    
    # Generazione embedding
    embeddings = compute_embeddings(
        X_train_txt, X_test_txt, 
        y_train, y_test,
        embeddings_file,
        model_name=DEFAULT_EMBEDDING_MODEL
    )
    
    if embeddings is None:
        raise RuntimeError("Errore: gli embedding non sono stati generati correttamente.")
    
    return embeddings


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(directory):
    """
    Crea una directory se non esiste.
    
    Args:
        directory (str): Percorso della directory
    """
    os.makedirs(directory, exist_ok=True)


def print_class_distribution(y, title="Distribuzione classi"):
    """
    Stampa la distribuzione delle classi.
    
    Args:
        y: Array o Series con le label
        title (str): Titolo da stampare
    """
    print(f"\n{Colors.BOLD}{title}:{Colors.END}")
    for label, count in Counter(y).items():
        print(f"  {label:<30} | {count:>5}")
    print()


# Assicurati che la directory embeddings esista
ensure_dir(EMBEDDINGS_DIR)
ensure_dir('models')
