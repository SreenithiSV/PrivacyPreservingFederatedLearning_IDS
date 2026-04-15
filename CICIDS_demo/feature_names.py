import pandas as pd
import numpy as np
import glob
import os

DATASET_DIR = r"C:\Users\SREENITHI\PrivacyPreservingFederatedLearning_IDS _HE\data"

csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))

# Load just ONE file (enough for column names)
df = pd.read_csv(csv_files[0])

# Clean column names
df.columns = df.columns.str.strip()

# Remove label column
feature_names = df.drop(columns=["Label"]).columns.tolist()

# Save
np.save("feature_names.npy", feature_names)

print(" feature_names.npy created successfully!")
print(f"Total features: {len(feature_names)}")