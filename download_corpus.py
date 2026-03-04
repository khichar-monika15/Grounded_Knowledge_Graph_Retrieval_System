"""Download and sample Enron email corpus."""
import pandas as pd
import os
import argparse


def download_corpus(sample_size: int = 200, output_path: str = "data/enron_sample.csv",
                    dataset_path: str = "dataset/Enron_emails.csv"):
    """Sample emails from peak Enron fraud period (2001)."""
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Total rows: {len(df)}")

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    print(f"Columns: {list(df.columns)}")

    # Parse dates
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Get body column
    body_col = 'body' if 'body' in df.columns else [c for c in df.columns if 'body' in c.lower()][0]
    sender_col = 'sender' if 'sender' in df.columns else [c for c in df.columns if 'from' in c.lower() or 'sender' in c.lower()][0]
    recipient_col = 'recipient' if 'recipient' in df.columns else [c for c in df.columns if 'to' in c.lower() or 'recip' in c.lower()][0]

    print(f"Using columns: date={date_col}, body={body_col}, sender={sender_col}, recipient={recipient_col}")

    # Filter to 2001 (peak fraud period)
    mask = (df[date_col] >= '2001-01-01') & (df[date_col] <= '2001-12-31')
    filtered = df[mask].dropna(subset=[body_col])
    print(f"Emails in 2001: {len(filtered)}")

    # Sample
    n = min(sample_size, len(filtered))
    sample = filtered.sample(n, random_state=42)

    # Standardize columns
    result = pd.DataFrame({
        'date': sample[date_col],
        'sender': sample[sender_col],
        'recipient': sample[recipient_col],
        'body': sample[body_col],
        'source_id': sample.index.astype(str),
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} emails to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--output", default="data/enron_sample.csv")
    parser.add_argument("--dataset", default="dataset/Enron_emails.csv")
    args = parser.parse_args()
    download_corpus(args.sample_size, args.output, args.dataset)
