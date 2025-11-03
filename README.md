# ML Project Work

Un framework modulare per esperimenti di Machine Learning su task di classificazione testuale, con focus su confronto di modelli e analisi delle learning curves.

## Indice

- [Panoramica](#panoramica)
- [Caratteristiche](#caratteristiche)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Struttura del Progetto](#struttura-del-progetto)
- [Utilizzo](#utilizzo)
  - [Test di Singoli Modelli](#test-di-singoli-modelli)
  - [Analisi Learning Curves](#analisi-learning-curves)
- [Modelli Supportati](#modelli-supportati)
- [Dataset](#dataset)
- [Output e Risultati](#output-e-risultati)
- [Utilità](#utilità)
- [API Python](#api-python)

---

## Panoramica

Questo progetto fornisce un'infrastruttura completa per:
- Generare embeddings testuali tramite **Sentence-BERT**
- Testare e confrontare diversi modelli di classificazione
- Analizzare le performance al variare della dimensione del training set (learning curves)
- Ottimizzare iperparametri tramite Grid Search con cross-validation

Il framework è pensato per essere **modulare**, **riproducibile** e **facilmente estendibile** con nuovi modelli o dataset.

---

## Caratteristiche

- **Embeddings Semantici**: Utilizzo di Sentence-BERT (`all-mpnet-base-v2`) per rappresentazioni dense del testo
- **Caching Intelligente**: Gli embeddings vengono calcolati una sola volta e salvati su disco
- **Grid Search Automatizzata**: Ottimizzazione degli iperparametri per ciascun modello
- **Metriche Multiple**: Supporto per accuracy, F1-macro, Cohen's Kappa
- **Learning Curves**: Analisi dettagliata delle performance al crescere dei dati di training
- **Visualizzazioni**: Grafici automatici delle learning curves per ogni metrica
- **Split Stratificato**: Mantiene la distribuzione delle classi in train e test set
- **Riproducibilità**: Seed fissi per garantire risultati riproducibili

---
 Stratificato: Mantiene la distribuzione delle classi in train e test set
Riproducibilità: Seed fissi per garantire risultati riproducibili
---

## Installazione

```bash
# Clona il repository
git clone https://github.com/edo-007/ml-projectwork.git
cd ml-projectwork

# Installa le dipendenze
pip install -r requirements.txt

# Crea le directory necessarie (opzionale, vengono create automaticamente)
mkdir -p embeddings models results
```

---

## Struttura del Progetto

```
ml-projectwork/
│
├── testmodel.py          # Script principale per testare singoli modelli
├── learning_curve.py     # Script per analisi learning curves
├── models.py             # Definizione modelli e griglie iperparametri
├── utils.py              # Funzioni di utilità (caricamento dati, embeddings)
├── clear.sh              # Script bash per pulizia file generati
│
├── embeddings/           # Cache embeddings (generati automaticamente)
├── models/               # Modelli addestrati salvati
├── results/              # Grafici e risultati degli esperimenti
│
└── data/dataset.csv           # I tuoi datasets 
```

---

## Utilizzo

### Test di Singoli Modelli

Il file `testmodel.py` permette di addestrare e valutare un singolo modello con Grid Search.

#### Esempio Base

```bash
# Test con Logistic Regression (default)
python testmodel.py --data dataset.csv
```

#### Parametri Disponibili

```bash
python testmodel.py \
    --data <percorso_dataset.csv>            # OBBLIGATORIO: dataset CSV
    --model <logreg|svm|dt|knn>              # Modello da usare (default: logreg)
    --refit <accuracy|f1_macro|cohen_kappa>  # Metrica per grid search (default: accuracy)
    --testsize <0.0-1.0>                     # Proporzione test set (default: 0.2)
    --seedsplit <int>                        # Seed per riproducibilità (default: 42)
```
#### Esempio con Parametri Personalizzati

```bash
# SVM con ottimizzazione su F1-macro
python testmodel.py --data data/dataset.csv \
    --model svm \
    --refit f1_macro \
    --testsize 0.3 \
    --seedsplit 42
```

---

### Analisi Learning Curves

Il file `learning_curve.py` analizza come le performance variano al crescere del training set.

#### Parametri Disponibili

```bash
python learning_curve.py \
    --data <percorso_dataset.csv>           # OBBLIGATORIO: dataset CSV
    --models <logreg svm dt knn>            # Modelli da confrontare (default: logreg svm)
    --train-sizes <0.1 0.2 ... 1.0>         # Dimensioni training set (default: varie)
    --refit <accuracy|f1_macro|cohen_kappa> # Metrica ottimizzazione (default: accuracy)
    --testsize <0.0-1.0>                    # Proporzione test set (default: 0.2)
    --seedsplit <int>                       # Seed per riproducibilità (default: 42)
    --output-dir <directory>                # Directory output (default: results)
    --no-plot                               # Non generare grafici
```
#### Esempio Base

```bash
# Confronto di tutti i modelli di default (logreg, svm)
python learning_curve.py --data dataset.csv
```

#### Esempio con Dimensioni Custom

```bash
# Test su dimensioni specifiche del training set (percentuali)
python learning_curve.py \
    --data dataset.csv \
    --models logreg svm \
    --train-sizes 0.1 0.2 0.3 0.5 0.7 1.0
```
Utilizzare `--no-plot` per il report senza generazione dei grafici


## Dataset

### Formato Richiesto

Il dataset deve essere un file CSV con **almeno** le seguenti colonne:

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `clean_text` | string | Testo preprocessato da classificare |
| `classificazione` | string/int | Etichetta della classe |

<!-- ### Esempio di Dataset -->
<!---->
<!-- ```csv -->
<!-- clean_text,classificazione -->
<!-- "questo è un esempio di testo positivo",positivo -->
<!-- "questo è un esempio di testo negativo",negativo -->
<!-- "testo neutro di esempio",neutro -->
<!-- ``` -->
## Output e Risultati

### Embeddings

Gli embeddings vengono salvati in `embeddings/` con il formato:

```
embeddings/
└── <nome_dataset>.joblib
```

Contenuto del file:
```python
{
    'X_train': ndarray,  # Shape: (n_samples_train, 768)
    'X_test': ndarray,   # Shape: (n_samples_test, 768)
    'y_train': ndarray,  # Shape: (n_samples_train,)
    'y_test': ndarray,   # Shape: (n_samples_test,)
    'metadata': {
        'model': 'all-mpnet-base-v2',
        'train_shape': (4000, 768),
        'test_shape': (1000, 768),
        'class_distribution_train': {...},
        'class_distribution_test': {...}
    }
}
```
