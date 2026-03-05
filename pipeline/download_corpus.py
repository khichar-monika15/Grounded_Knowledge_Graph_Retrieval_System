"""Download and sample Enron email corpus.

Uses chunked CSV reading (chunksize rows at a time) to avoid loading the full
918 MB dataset into memory. Only 2001 rows are kept; they are accumulated across
chunks before sampling.
"""
import argparse
import os

import pandas as pd


def download_corpus(sample_size: int = 200, output_path: str = "data/enron_sample.csv",
                    dataset_path: str = "dataset/Enron_emails.csv",
                    chunksize: int = 10_000):
    """Sample emails from peak Enron fraud period (2001).

    Args:
        sample_size: Number of emails to sample (max 500 recommended).
        output_path: Where to write the sampled CSV.
        dataset_path: Path to the full Enron CSV (~918 MB).
        chunksize: Rows per chunk during CSV reading (controls peak memory use).
    """
    print(f"Loading {dataset_path} in {chunksize}-row chunks...")
    accumulated: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(dataset_path, chunksize=chunksize):
        total_rows += len(chunk)
        chunk.columns = chunk.columns.str.lower().str.strip()

        date_col = 'date' if 'date' in chunk.columns else chunk.columns[0]
        body_col = next((c for c in chunk.columns if 'body' in c.lower()), None)
        if body_col is None:
            continue

        chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
        mask = (chunk[date_col] >= '2001-01-01') & (chunk[date_col] <= '2001-12-31')
        filtered = chunk[mask].dropna(subset=[body_col])
        if not filtered.empty:
            accumulated.append(filtered)

    print(f"Scanned {total_rows:,} rows total.")

    if not accumulated:
        raise ValueError("No 2001 emails found — check dataset path and column names.")

    df = pd.concat(accumulated, ignore_index=True)
    print(f"2001 emails found: {len(df)}")

    # Resolve column names from the concatenated frame
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    body_col = next(c for c in df.columns if 'body' in c.lower())
    sender_col = next(
        (c for c in df.columns if 'sender' in c.lower() or 'from' in c.lower()), df.columns[1]
    )
    recipient_col = next(
        (c for c in df.columns if 'recipient' in c.lower() or c == 'to'), df.columns[2]
    )
    print(f"Columns: date={date_col}, sender={sender_col}, recipient={recipient_col}, body={body_col}")

    n = min(sample_size, len(df))
    sample = df.sample(n, random_state=42)

    result = pd.DataFrame({
        'date': sample[date_col],
        'sender': sample[sender_col],
        'recipient': sample[recipient_col],
        'body': sample[body_col],
        'source_id': sample.index.astype(str),
    })

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} emails to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Enron emails for pipeline")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--output", default="data/enron_sample.csv")
    parser.add_argument("--dataset", default="dataset/Enron_emails.csv")
    args = parser.parse_args()
    download_corpus(args.sample_size, args.output, args.dataset)
