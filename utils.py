import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split 
from collections import Counter


class Colors:
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


EMBEDDINGS_DIR = 'embeddings'

    
def compute_embeddings(train_texts, test_texts, y_train, y_test, embeddings_file, model_name='all-mpnet-base-v2'):
    """
    Genera gli embedding per train e test set e li salva insieme alle label.
    
    Args:
        train_texts: Lista di testi per il training
        test_texts: Lista di testi per il test
        y_train: Label del training set
        y_test: Label del test set
        embeddings_file: Percorso del file dove salvare gli embedding
        model_name: Nome del modello SentenceTransformer da usare
    
    Returns:
        Dictionary contenente embedding e label
    """
    print(f"Download del modello SBERT: {model_name}...")
    model = SentenceTransformer(model_name)

    print("Generazione degli embedding...")
    try:
        train_embeddings = model.encode(train_texts, show_progress_bar=True)
        test_embeddings = model.encode(test_texts, show_progress_bar=True)
        
        embeddings = {
            'X_train': train_embeddings,
            'X_test':  test_embeddings,
            'y_train': y_train.values,
            'y_test':  y_test.values,
            'metadata': {
                'model': model_name,
                'train_shape': train_embeddings.shape,
                'test_shape': test_embeddings.shape,
                'class_distribution_train': pd.Series(y_train).value_counts().to_dict(),
                'class_distribution_test': pd.Series(y_test).value_counts().to_dict()
            }
        }
    except Exception as e:
        print(f"Errore durante la generazione degli embedding: {e}")
        return None

    joblib.dump(embeddings, embeddings_file)
    print("Embedding salvati in " + Colors.GREEN + f"{embeddings_file}" + Colors.END)
    print(f"Shape train: {embeddings['X_train'].shape}, Shape test: {embeddings['X_test'].shape}")

    return embeddings





def validate_dataset(df):
    """
    Valida che il dataset contenga le colonne necessarie.
    
    Args:
        df: DataFrame da validare
    
    Raises:
        ValueError: Se mancano colonne richieste
    """
    required_cols = ['clean_text', 'classificazione']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")

    print(f"{len(df)} righe, {len(df.columns)} colonne")
    conteggi = df['classificazione'].value_counts()
    for chiave, valore in conteggi.items():
        print(f"Classificazione: {chiave:<20} | #rows: {valore:<5}")

    print("\n")



def load_embeddings(embeddings_file):
    
    print(Colors.BOLD + Colors.GREEN + f"Caricamento embedding da: {embeddings_file}" + Colors.END)
    embeddings = joblib.load(embeddings_file)

    # Verifica integrità
    required_keys = ['X_train', 'X_test', 'y_train', 'y_test']
    if all(key in embeddings for key in required_keys):

        print("Embedding caricati correttamente!")
        if 'metadata' in embeddings:
            print("Metadati embedding:")
            for key, value in embeddings['metadata'].items():
                print(f"  {key}: {value}")

    else:
        embeddings = None
        
    return embeddings



def get_embeddings(args):
    """
    Funzione principale per l'addestramento del classificatore KNN.
    """
    # CONTROLLO PRIORITARIO: verifica se gli embedding esistono già
    
    # Se gli embedding non sono stati caricati, procedi con il dataset

        # Validazione path dataset
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset non trovato: {args.data}")
    
    print(Colors.BOLD + f"\nCaricamento dataset da: {args.data}" + Colors.END)
    try: df = pd.read_csv(args.data)
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento del dataset: {e}")

    validate_dataset(df)

    # Sampling bilanciato se richiesto
    print(f"Dataset finale: {len(df):>10} righe")

    # Preparazione dati
    X = df['clean_text'].astype(str).tolist()
    y = df['classificazione']

    # Split train/test
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X, y, test_size=args.testsize, random_state=args.seedsplit, stratify=y
    )

    print(f"Train: {len(X_train_txt):>19} campioni")
    print(f"Test: {len(X_test_txt):>19} campioni  ({args.testsize} %)\n")

    # Distribuzione classi
    print("Distribuzione classi nel TRAIN:")
    print({label: count for label, count in Counter(y_train).items()})
    print("Distribuzione classi nel TEST:")
    print({label: count for label, count in Counter(y_test).items()})
    print("\n")

    
    # Generazione embedding
    embeddings_file = f"{EMBEDDINGS_DIR}/{Path(args.data).stem}.joblib"
    embeddings = compute_embeddings(
        X_train_txt, X_test_txt, y_train, y_test, 
        embeddings_file, model_name='all-mpnet-base-v2'
    )
    if embeddings is None:
        raise RuntimeError("Errore: gli embedding non sono stati generati correttamente.")
    
    print("\n")
    print(Colors.BOLD + Colors.PURPLE + f"Avvio Grid Search per {args.model.upper()}..." + Colors.END)
    
    # return embeddings
    return embeddings 
    


