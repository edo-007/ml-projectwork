#!/bin/bash

# Script per eliminare tutti i file .joblib nelle cartelle embeddings/ e results/

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Pulizia file .joblib${NC}"
echo -e "${BLUE}========================================${NC}"

# Contatori
count_embeddings=0
count_results=0
total_size=0

# Funzione per calcolare la dimensione totale
calculate_size() {
    local dir=$1
    if [ -d "$dir" ]; then
        size=$(find "$dir" -name "*.joblib" -type f -exec du -cb {} + 2>/dev/null | grep total$ | cut -f1)
        echo ${size:-0}
    else
        echo 0
    fi
}

# Calcola dimensione prima della pulizia
echo -e "\n${YELLOW}Analisi in corso...${NC}"
size_embeddings=$(calculate_size "embeddings")
size_results=$(calculate_size "results")
total_size=$((size_embeddings + size_results))

# Mostra cosa verrà eliminato
if [ -d "embeddings" ] && [ "$1" != "--ee" ]; then
    count_embeddings=$(find embeddings -name "*.joblib" -type f 2>/dev/null | wc -l)
    if [ $count_embeddings -gt 0 ]; then
        echo -e "\n${YELLOW}File in embeddings/:${NC}"
        find embeddings -name "*.joblib" -type f -exec ls -lh {} \; | awk '{print "  - " $9 " (" $5 ")"}'
    fi
fi

if [ -d "results" ]; then
    count_results=$(find results -type f \( -name "*.json" -o -name "*.csv" \)  2>/dev/null | wc -l)
    if [ $count_results -gt 0 ]; then
        echo -e "\n${YELLOW}File in results/:${NC}"
        find results -name "*.json" -type f -exec ls -lh {} \; | awk '{print "  - " $9 " (" $5 ")"}'
        find results -name "*.csv" -type f -exec ls -lh {} \; | awk '{print "  - " $9 " (" $5 ")"}'
    fi
fi



total_files=$((count_embeddings + count_results))

# Se non ci sono file da eliminare
if [ $total_files -eq 0 ]; then
    echo -e "\n${GREEN}Nessun file .joblib trovato. Le cartelle sono già pulite!${NC}"
    exit 0
fi

# Mostra riepilogo
echo -e "\n${YELLOW}Riepilogo:${NC}"
echo -e "  File in embeddings/: ${count_embeddings}"
echo -e "  File in results/:     ${count_results}"
echo -e "  Totale file:         ${total_files}"
if [ $total_size -gt 0 ]; then
    size_mb=$(echo "scale=2; $total_size / 1048576" | bc)
    echo -e "  Spazio liberato:     ${size_mb} MB"
fi

# Chiedi conferma
echo -e "\n${RED}Attenzione: questa operazione è irreversibile!${NC}"
read -p "Vuoi procedere con l'eliminazione? (s/N): " confirm

if [[ $confirm == [sS] || $confirm == [sS][iI] ]]; then
    echo -e "\n${YELLOW}Eliminazione in corso...${NC}"
    
    # Elimina file da embeddings/
    if [ -d "embeddings" ] && [ $count_embeddings -gt 0 ]; then
        find embeddings -name "*.joblib" -type f -delete
        echo -e "${GREEN}✓ Eliminati $count_embeddings file da embeddings/${NC}"
    fi
    
    # Elimina file da results/
    if [ -d "results" ] && [ $count_results -gt 0 ]; then
        find results -name "*.json" -type f -delete
        find results -name "*.csv" -type f -delete
        echo -e "${GREEN}✓ Eliminati $count_results file da results/${NC}"
    fi
    
    echo -e "\n${GREEN}Pulizia completata con successo!${NC}"
    
else
    echo -e "\n${BLUE}Operazione annullata.${NC}"
    exit 0
fi

echo -e "${BLUE}========================================${NC}"
