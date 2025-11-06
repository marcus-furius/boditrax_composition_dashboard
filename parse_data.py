"""
Quick data parser to understand Boditrax CSV structure
"""
import pandas as pd
from datetime import datetime
from dateutil import parser as date_parser

# Parse the multi-section CSV
with open('data/BoditraxAccount_20251006_102227.csv', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

# Find section boundaries
scan_start = None
for i, line in enumerate(lines):
    if line.strip() == 'User Scan Details':
        scan_start = i + 2  # Skip header row
        break

# Extract scan data
scan_data = []
for i in range(scan_start, len(lines)):
    line = lines[i].strip()
    if not line or line == 'User Login Details':
        break
    parts = line.split(',')
    if len(parts) == 3:
        # Replace non-breaking space with regular space
        date_str = parts[2].replace('\u202f', ' ')
        scan_data.append({
            'Metric': parts[0],
            'Value': parts[1],
            'Date': date_str
        })

df = pd.DataFrame(scan_data)
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df['Date'] = df['Date'].apply(lambda x: date_parser.parse(x))

# Get unique dates (scan sessions)
unique_dates = df['Date'].dt.date.unique()
print(f"Total scan sessions: {len(unique_dates)}")
print(f"Date range: {unique_dates.min()} to {unique_dates.max()}")
print(f"\nUnique metrics tracked ({len(df['Metric'].unique())}):")
for metric in sorted(df['Metric'].unique()):
    print(f"  - {metric}")

# Show metrics per scan
print(f"\nMetrics per scan session: {df.groupby(df['Date'].dt.date).size().mean():.0f}")
