import os
import time
import pandas as pd
import boto3
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse

def read_single_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Erreur lors de la lecture de {filepath}: {e}")
        return pd.DataFrame()


def unpack_data(input_dir, bucket_name, output_file_name, max_workers=4):
    """
    Combine multiple CSV files and upload to S3.

    Parameters
    ----------
    input_dir : str
    Directory containing subdirectories with CSV files.
    bucket_name : str
    Name of the S3 bucket for raw data.
    output_file_name : str
    Name of the combined CSV file in S3.
    max_workers : int
    Number of threads for parallel reading.

    Steps
    -----
    1. Walk input_dir to collect all .csv file paths
    2. Read files SEQUENTIALLY, measure time
    3. Read files in PARALLEL with ThreadPoolExecutor, measure time

    4. Print both times and the speedup
    5. Concatenate all DataFrames
    6. Upload the combined CSV to the S3 bucket
    """
    s3 = boto3.client('s3', endpoint_url='http://localhost:4566')
    input_path = Path(input_dir)
    # Step 1: collect all CSV file paths
    csv_files = [
        p for p in input_path.rglob("*") 
        if p.is_file() and "Zone.Identifier" not in p.name
    ]

    # Step 2: sequential reading + timing
    start = time.perf_counter()
    sequential_dfs = []
    for file in csv_files:
        df = read_single_csv(file)
        if not df.empty:
            sequential_dfs.append(df)
    t_sequential = time.perf_counter() - start

    # Step 3: parallel reading + timing
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map applique la fonction à toute la liste en parallèle
        parallel_dfs = list(executor.map(read_single_csv, csv_files))
        
    # Filtrer les DataFrames vides en cas d'erreur sur un fichier
    parallel_dfs = [df for df in parallel_dfs if not df.empty]
    t_parallel = time.perf_counter() - start

    # Step 4: print comparison
    print(f"Sequentiel : {t_sequential:.2f}s")
    print(f"Parallele : {t_parallel:.2f}s")
    print(f"Speedup : {t_sequential/t_parallel:.1f}x")

    # Step 5: concatenate
    print("Étape 5 : Concaténation des données...")
    final_df = pd.concat(parallel_dfs, ignore_index=True)
    print(f"Nombre total de lignes combinées : {len(final_df)}")

    # Étape 6 : Conversion en mémoire et téléversement direct vers S3 (LocalStack)
    print(f"Étape 6 : Téléversement vers le bucket S3 '{bucket_name}'...")
    
    # On convertit le DataFrame en string CSV dans un buffer en mémoire (StringIO)
    csv_buffer = io.StringIO()
    final_df.to_csv(csv_buffer, index=False)
    
    # s3.put_object attend des bytes, on encode donc la chaîne de caractères
    s3.put_object(
        Bucket=bucket_name,
        Key=output_file_name,
        Body=csv_buffer.getvalue().encode('utf-8')
    )
    
    print(f"Le fichier '{output_file_name}' a été injecté dans la zone Raw.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack and combine CSV files.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--output_file_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=4)

    args = parser.parse_args()

    unpack_data(
        input_dir=args.input_dir,
        bucket_name=args.bucket_name,
        output_file_name=args.output_file_name,
        max_workers=args.max_workers
    )