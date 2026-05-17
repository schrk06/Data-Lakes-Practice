import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def preprocess_data(data_file: str, output_dir: str) -> None:
    """
    Preprocess raw protein sequence data for model training.

    This function loads the raw data, cleans it, encodes labels, and splits
    it into train/validation/test sets. The split strategy must handle the
    extreme class imbalance in the Pfam dataset.

    Parameters
    ----------
    data_file : str
        Path to the combined raw data CSV file.
    output_dir : str
        Directory where processed files will be saved.

    Steps
    -----
    1. Load the data with pandas
    2. Remove rows with missing values
    3. Encode the 'family_accession' column with LabelEncoder
    4. Design and implement a split strategy that handles class imbalance
    5. Save train.csv, val.csv, and test.csv to output_dir

    Notes
    -----
    sklearn's train_test_split with stratify will fail on this dataset
    because some classes have only one sample. You need to implement
    a custom strategy.
    """
    data_path = Path(data_file)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des données
    df = pd.read_csv(data_path)

    # 2. Nettoyage
    df = df.dropna()

    # 3. Encodage des labels
    le = LabelEncoder()
    df['family_accession_encoded'] = le.fit_transform(df['family_accession'])

    # 4. Stratégie de fractionnement sur-mesure
    print("Séparation des données en cours...")
    
    # On mélange tout au début pour garantir l'aléatoire
    df = shuffle(df, random_state=42).reset_index(drop=True)

    train_list, val_list, test_list = [], [], []

    grouped = df.groupby('family_accession_encoded')

    for _, group in grouped:
        n = len(group)
        if n == 1:
            # 1 élément -> Train
            train_list.append(group)
        elif n == 2:
            # 2 éléments -> 1 Train, 1 Test
            train_list.append(group.iloc[[0]])
            test_list.append(group.iloc[[1]])
        elif n == 3:
            # 3 éléments -> 1 Train, 1 Val, 1 Test
            train_list.append(group.iloc[[0]])
            val_list.append(group.iloc[[1]])
            test_list.append(group.iloc[[2]])
        else:
            # 4 éléments ou plus -> Calcul des proportions 70/15/15
            # max(1, ...) garantit qu'on a toujours au moins 1 élément dans Val et Test
            n_val = max(1, int(n * 0.15))
            n_test = max(1, int(n * 0.15))
            
            val_list.append(group.iloc[:n_val])
            test_list.append(group.iloc[n_val : n_val + n_test])
            train_list.append(group.iloc[n_val + n_test:])

    # Concaténation pour recréer les DataFrames finaux
    train_final = pd.concat(train_list, ignore_index=True)
    val_final = pd.concat(val_list, ignore_index=True)
    test_final = pd.concat(test_list, ignore_index=True)

    # Un dernier mélange pour s'assurer que les classes ne sont pas triées
    train_final = shuffle(train_final, random_state=42)
    val_final = shuffle(val_final, random_state=42)
    test_final = shuffle(test_final, random_state=42)

    # 5. Sauvegarde
    train_final.to_csv(output_path / "train.csv", index=False)
    val_final.to_csv(output_path / "val.csv", index=False)
    test_final.to_csv(output_path / "test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Pfam data.")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)