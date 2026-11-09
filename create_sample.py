import pandas as pd

df = pd.read_csv('Reviews.csv')
sample_df = df[['Text']].dropna().head(5000)
sample_df.to_csv('Reviews_sample.csv', index=False)

print("✅ Reviews_sample.csv created!")
