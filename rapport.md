# Documentation du Projet : Pipeline DVC et Data Engineering

Ce document détaille l'architecture, l'utilisation et les optimisations mises en place pour le traitement massif de séquences protéiques (dataset PFAM).

---

# 1. Architecture et Concepts Fondamentaux

L'infrastructure repose sur la séparation stricte entre le versionnement du code (Git), le stockage des données massives (S3) et l'orchestration des traitements (DVC).

## Bucket S3 (Simple Storage Service)

Un bucket est un conteneur logique de stockage d'objets.
Ici, AWS S3 est émulé localement par **LocalStack**.

L'architecture suit le standard **Medallion** (Data Lake multi-zones) :

* `raw` : reçoit les données brutes extraites de la source.
* `staging` : contient les données nettoyées, filtrées et séparées (train/val/test).
* `curated` : stocke les données finalisées, transformées en tenseurs ou tokens, prêtes à être ingérées par un modèle de Machine Learning.
* `dvc-store` : espace technique de DVC.

---

## DVC (Data Version Control)

DVC est un système de contrôle de version conçu pour le Machine Learning.

Git étant incapable de gérer efficacement des fichiers CSV de plusieurs gigaoctets, DVC remplace les données locales par de petits fichiers de métadonnées (`dvc.lock`) contenant des pointeurs cryptographiques (hachages) vers les vraies données stockées sur S3.

---

## DAG (Directed Acyclic Graph)

Un graphe orienté acyclique définit un flux de travail unidirectionnel.

Dans le fichier `dvc.yaml`, les étapes :

```text
unpack ➔ preprocess ➔ curate
```

forment un DAG.

Si une donnée source change, DVC parcourt le graphe pour ne réexécuter **que** les nœuds impactés.

---

# 2. Manuel d'Utilisation

## A. Démarrage de l'environnement

À chaque redémarrage de la machine, l'infrastructure locale et l'environnement virtuel doivent être réactivés.

```bash
# 1. Activation de l'environnement virtuel Python
source .venv/bin/activate

# 2. Démarrage du conteneur LocalStack (serveur S3)
export LOCALSTACK_AUTH_TOKEN="votre_token"

localstack start -d
```

---

## B. Exécution du Pipeline

La commande d'orchestration unique lance l'analyse du DAG.

DVC vérifie les hachages :

* si une modification est détectée, le pipeline est réexécuté ;
* sinon, DVC réutilise les artefacts présents dans son cache.

```bash
dvc repro
```

---

## C. Versionnement de l'état

Une fois le pipeline exécuté avec succès, il faut figer cet état dans Git.

Git versionne :

* le code source ;
* les fichiers de configuration DVC ;
* les métadonnées du pipeline.

```bash
# Ajout des fichiers de configuration DVC
git add dvc.yaml dvc.lock .dvc/config

# Création d'un commit
git commit -m "Mise à jour du pipeline de traitement de données"

# Envoi vers le dépôt distant
git push origin nom_de_la_branche
```

---

# 3. Mécanismes d'Optimisation et Code Clé

Le projet intègre trois niveaux d'optimisation afin de traiter des millions de lignes beaucoup plus rapidement.

---

## Optimisation 1 : Parallélisme I/O (Multi-threading)

### Problème

Lire des milliers de petits fichiers CSV un par un bloque le CPU, qui attend les opérations disque (*I/O bound*).

### Solution

Utilisation de `ThreadPoolExecutor` pour lire plusieurs fichiers simultanément.

### Fichier : `src/unpack_data.py`

```python
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Lecture asynchrone :
# le CPU ne reste pas inactif pendant les chargements disque.
with ThreadPoolExecutor() as executor:
    dataframes = list(executor.map(pd.read_csv, csv_files))

combined_df = pd.concat(dataframes, ignore_index=True)
```

---

## Optimisation 2 : Compilation JIT et Algorithme (Numba)

### Problème

Boucler sur des DataFrames Pandas avec des conditions pour effectuer un split `(train/val/test)` sur des millions de lignes est extrêmement lent en Python natif.

### Solution

Deux optimisations sont combinées :

1. Tri préalable des données (`O(n log n)`) pour regrouper les classes.
2. Compilation du code Python en code machine via **Numba** avec le décorateur `@njit`.

### Fichier : `src/preprocess_to_staging.py`

```python
from numba import njit
import numpy as np

# @njit compile la fonction.
# L'exécution se fait à une vitesse proche du C.
@njit
def assign_splits(group_starts, group_sizes, n_samples):

    assignments = np.zeros(n_samples, dtype=np.int64)

    for g in range(len(group_starts)):

        start = group_starts[g]
        size = group_sizes[g]

        # Logique de distribution ultra-rapide
        if size == 1:
            assignments[start] = 2

    return assignments
```

---

## Optimisation 3 : Parallélisme Matériel (Batching de Tokens)

### Problème

Encoder les séquences protéiques une par une est inefficace.

### Solution

Traitement par lots (*batching*) afin d'utiliser le parallélisme interne du tokenizer Hugging Face (backend Rust).

### Fichier : `src/process_to_curated.py`

```python
# Traitement par lot pour exploiter le parallélisme interne du backend Rust

all_input_ids = []
batch_size = 512

for i in range(0, len(sequences), batch_size):

    batch = sequences[i:i + batch_size]

    tokenized = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512
    )

    all_input_ids.extend(tokenized["input_ids"])
```

---

# Conclusion

Cette architecture permet :

* un traitement scalable de datasets massifs ;
* un versionnement propre des données et des pipelines ;
* une reproductibilité complète des expériences ;
* des gains de performance importants grâce :

  * au parallélisme I/O ;
  * à la compilation JIT ;
  * au batching et au parallélisme matériel.

Le pipeline devient ainsi robuste, reproductible et adapté aux workflows modernes de Machine Learning et Data Engineering.
