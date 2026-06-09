import pandas as pd
import matplotlib.pyplot as plt

from utils import load_dataset

df = load_dataset()

print("=" * 50)
print("DATASET INFO")
print("=" * 50)

print(df.shape)

print(df.columns)

print(df.isnull().sum())

df["method_length"] = df["body"].fillna("").str.len()

df["javadoc_length"] = (
    df["javaDocFirstSentence"]
    .fillna("")
    .str.len()
)

print(df["method_length"].describe())

print(df["javadoc_length"].describe())

plt.hist(df["method_length"], bins=100)

plt.title("Method Length")

plt.show()

plt.hist(df["javadoc_length"], bins=50)

plt.title("JavaDoc Length")

plt.show()