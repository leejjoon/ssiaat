import pandas as pd
import os
import time
from ssiaat.core import find_latest_uri

def main():
    # 1. Read the ECSV file using pandas
    file_path = "eso_244.ecsv"
    print(f"Reading {file_path}...")
    
    raw_df = pd.read_csv(file_path, comment='#', sep=' ')
    
    # 2. Get unique filenames
    filenames = raw_df['filename'].unique()
    print(f"Found {len(filenames)} unique filenames.")
    
    # 3. Check for availability on S3 using the package
    root_uri = "s3://nasa-irsa-spherex"
    release = "qr2"
    
    print(f"\nChecking for latest URIs on S3 (Concurrent/Async): {root_uri}")
    
    start_time = time.time()
    # Now using the high-performance async implementation behind the scenes
    latest_uris = find_latest_uri(filenames, root_uri, release=release, progress=True, max_concurrency=30)
    end_time = time.time()
    
    # 4. Show results
    results = pd.DataFrame({
        'filename': filenames,
        's3_uri': latest_uris
    })
    
    print("\nAvailability check results (first 5):")
    print(results.head(5))
    
    found_count = latest_uris.notna().sum()
    print(f"\nSuccessfully found {found_count} out of {len(filenames)} unique files on S3.")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
