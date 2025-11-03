# ML Project Work

Un framework modulare per esperimenti di Machine Learning su task di classificazione testuale, con focus su confronto di modelli e analisi delle learning curves.

## 📑 Indice

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

## Requisiti

```bash
requirements.txt
```

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

