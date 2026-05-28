import argparse
import io
import json
import time
import boto3
from concurrent.futures import ThreadPoolExecutor
from numba import njit
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@njit
def assign_splits(group_starts, group_sizes, n_samples):
    """
    Assign each sample to train(0), val(1), or test(2).

    Parameters
    ----------
    group_starts : np.ndarray[int64]
        Start index of each class in the sorted array.
    group_sizes : np.ndarray[int64]
        Number of samples in each class.
    n_samples : int
        Total number of samples.

    Returns
    -------
    assignments : np.ndarray[int64]
        0=train, 1=val, 2=test for each sample.

    Rules
    -----
    - 1 sample -> test (2)
    - 2 samples -> val (1), test (2)
    - 3 samples -> train (0), val (1), test (2)
    - 4+ samples -> 80% train, 10% val, 10% test (approx.)
    """
    assignments = np.zeros(n_samples, dtype=np.int64)
    n_groups = len(group_starts)
    
    for g in range(n_groups):
        start = group_starts[g]
        size = group_sizes[g]
        
        if size == 1:
            assignments[start] = 2
        elif size == 2:
            assignments[start] = 1
            assignments[start + 1] = 2
        elif size == 3:
            assignments[start] = 0
            assignments[start + 1] = 1
            assignments[start + 2] = 2
        else:
            # Règle des 4+ : calcul des tailles proportionnelles
            n_val = int(round(size * 0.1))
            n_test = int(round(size * 0.1))
            
            # Garantir au moins 1 échantillon par jeu si non nul après arrondi
            if n_val == 0:
                n_val = 1
            if n_test == 0:
                n_test = 1
                
            n_train = size - n_val - n_test
            
            # Remplissage par tranches d'indices relatives au groupe
            for i in range(n_train):
                assignments[start + i] = 0
            for i in range(n_train, n_train + n_val):
                assignments[start + i] = 1
            for i in range(n_train + n_val, size):
                assignments[start + i] = 2
                
    return assignments


def assign_splits_naive(group_starts, group_sizes, n_samples):
    """Même logique algorithmique sans décoration @njit pour analyse comparative."""
    assignments = np.zeros(n_samples, dtype=np.int64)
    n_groups = len(group_starts)
    for g in range(n_groups):
        start = group_starts[g]
        size = group_sizes[g]
        if size == 1:
            assignments[start] = 2
        elif size == 2:
            assignments[start] = 1
            assignments[start + 1] = 2
        elif size == 3:
            assignments[start] = 0
            assignments[start + 1] = 1
            assignments[start + 2] = 2
        else:
            n_val = int(round(size * 0.1))
            n_test = int(round(size * 0.1))
            if n_val == 0:
                n_val = 1
            if n_test == 0:
                n_test = 1
            n_train = size - n_val - n_test
            for i in range(n_train):
                assignments[start + i] = 0
            for i in range(n_train, n_train + n_val):
                assignments[start + i] = 1
            for i in range(n_train + n_val, size):
                assignments[start + i] = 2
    return assignments


def preprocess_to_staging(bucket_raw, bucket_staging, input_file, output_prefix):
    """
    Download raw data, preprocess with numba-accelerated split,
    upload results to staging.
    """
    s3 = boto3.client('s3', endpoint_url='http://localhost:4566')

    # Step 1: download from S3
    print("Step 1: Downloading data from S3...")
    response = s3.get_object(Bucket=bucket_raw, Key=input_file)
    data = pd.read_csv(io.BytesIO(response['Body'].read()))

    # Step 2: clean
    print("Step 2: Cleaning data...")
    data = data.dropna()

    # Step 3: encode labels
    print("Step 3: Encoding labels...")
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(data['family_accession'])

    # Step 4: sort by class (key algorithmic optimization!)
    print("Step 4: Sorting data by encoded class...")
    data_sorted = data.sort_values('class_encoded').reset_index(drop=True)

    # Step 5: compute group boundaries
    print("Step 5: Computing group boundaries...")
    class_ids = data_sorted['class_encoded'].values
    unique_classes, counts = np.unique(class_ids, return_counts=True)
    starts = np.cumsum(counts) - counts
    starts = starts.astype(np.int64)
    counts = counts.astype(np.int64)

    # Step 6: numba-accelerated split
    # Warm-up (first call compiles the function)
    _ = assign_splits(starts[:10], counts[:10], int(counts[:10].sum()))

    start_time = time.perf_counter()
    assignments = assign_splits(starts, counts, len(data_sorted))
    t_numba = time.perf_counter() - start_time
    print(f"Numba split : {t_numba:.3f}s")

    # Step 7 (optional): compare with naive Python timing
    print("Step 7: Running naive Python split for benchmarking...")
    start_naive = time.perf_counter()
    _ = assign_splits_naive(starts, counts, len(data_sorted))
    t_naive = time.perf_counter() - start_naive
    print(f"Naive Python split : {t_naive:.3f}s")
    print(f"Speedup de Numba : {t_naive / t_numba:.1f}x")

    # Step 8: split DataFrame
    print("Step 8: Splitting DataFrames...")
    train_data = data_sorted[assignments == 0].drop(
        columns=["family_id", "sequence_name", "family_accession"])
    dev_data = data_sorted[assignments == 1].drop(
        columns=["family_id", "sequence_name", "family_accession"])
    test_data = data_sorted[assignments == 2].drop(
        columns=["family_id", "sequence_name", "family_accession"])

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

    # Step 9: upload to staging
    print("Step 9: Uploading preprocessed datasets to staging...")
    for name, df in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        s3.put_object(Bucket=bucket_staging,
                      Key=f"{output_prefix}_{name}.csv",
                      Body=buf.getvalue().encode('utf-8'))
        print(f"{name} uploaded to staging.")

    # Upload label_mapping and class_weights as in TP1
    print("Uploading metadata (label mapping and class weights)...")
    
    # Construction du dictionnaire d'encodage
    label_mapping = {int(code): label for code, label in enumerate(label_encoder.classes_)}
    label_mapping_buf = io.StringIO()
    json.dump(label_mapping, label_mapping_buf)
    s3.put_object(Bucket=bucket_staging,
                  Key=f"{output_prefix}_label_mapping.json",
                  Body=label_mapping_buf.getvalue().encode('utf-8'))
                  
    # Calcul et normalisation des poids des classes
    total_samples = len(train_data)
    class_counts = train_data['class_encoded'].value_counts()
    num_classes = len(unique_classes)
    
    # Formule standard : poids = total_samples / (num_classes * count)
    class_weights = {}
    for cls_idx, count in class_counts.items():
        class_weights[int(cls_idx)] = float(total_samples / (num_classes * count))
        
    class_weights_buf = io.StringIO()
    json.dump(class_weights, class_weights_buf)
    s3.put_object(Bucket=bucket_staging,
                  Key=f"{output_prefix}_class_weights.json",
                  Body=class_weights_buf.getvalue().encode('utf-8'))
                  
    print("Metadata uploaded to staging. 🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split genomic data using Numba.")
    parser.add_argument("--bucket_raw", type=str, default="raw")
    parser.add_argument("--bucket_staging", type=str, default="staging")
    parser.add_argument("--input_file", type=str, default="combined_raw.csv")
    parser.add_argument("--output_prefix", type=str, default="preprocessed")

    args = parser.parse_args()
    
    preprocess_to_staging(
        bucket_raw=args.bucket_raw,
        bucket_staging=args.bucket_staging,
        input_file=args.input_file,
        output_prefix=args.output_prefix
    )