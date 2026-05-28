import argparse
import io
import time
import pandas as pd
import boto3
from transformers import AutoTokenizer

def process_to_curated(bucket_staging, bucket_curated, input_file, output_file, batch_size=512):
    s3 = boto3.client('s3', endpoint_url='http://localhost:4566')
    
    # Etape 1: Téléchargement
    response = s3.get_object(Bucket=bucket_staging, Key=input_file)
    data = pd.read_csv(io.BytesIO(response['Body'].read()))
    sequences = data['sequence'].tolist()

    # Etape 2: Chargement du tokenizer ESM-2
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # Etape 3: Tokenisation séquentielle (mesure)
    subset = sequences[:1000]
    start = time.perf_counter()
    for seq in subset:
        tokenizer(seq, padding=True, truncation=True, max_length=512)
    t_seq = time.perf_counter() - start

    # Etape 4: Tokenisation par batch (mesure)
    start = time.perf_counter()
    tokenizer(subset, padding=True, truncation=True, max_length=512)
    t_batch = time.perf_counter() - start

    # Etape 5: Comparaison
    print(f"Sequentiel (1000 seq) : {t_seq:.2f}s")
    print(f"Batch (1000 seq)      : {t_batch:.2f}s")
    print(f"Speedup               : {t_seq/t_batch:.1f}x")

    # Etape 6: Tokenisation du dataset complet par lots
    print("Tokenisation du dataset complet...")
    all_input_ids = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        tokenized = tokenizer(batch, padding=True, truncation=True, max_length=512)
        all_input_ids.extend(tokenized['input_ids'])

    # Etape 7: Ajout au DataFrame
    data['input_ids'] = all_input_ids

    # Etape 8: Téléversement vers Curated
    buf = io.StringIO()
    data.to_csv(buf, index=False)
    s3.put_object(
        Bucket=bucket_curated, 
        Key=output_file, 
        Body=buf.getvalue().encode('utf-8')
    )
    print(f"Fichier {output_file} exporté vers s3://{bucket_curated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_staging", type=str, required=True)
    parser.add_argument("--bucket_curated", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    process_to_curated(args.bucket_staging, args.bucket_curated, args.input_file, args.output_file)