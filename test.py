import pandas as pd

path = "./data/STE/raw/mathematics/filtered.csv"

df = pd.read_csv(path)

total_rows = len(df)
unique_users = df["UserId"].nunique()
ratio = unique_users / total_rows if total_rows > 0 else float("nan")

print(f"File: {path}")
print(f"Total rows (posts): {total_rows}")
print(f"Unique UserId: {unique_users}")
print(f"Unique UserId / total rows: {ratio:.6f}  ({ratio*100:.2f}%)")

unique_