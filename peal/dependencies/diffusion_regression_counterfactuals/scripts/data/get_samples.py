import pandas as pd

df = pd.read_csv("imdb_clean_atleast256.csv")

# Sample 2 images of each unique age value
df = (
    df.groupby("age")
    .filter(lambda x: len(x) >= 2)
    .groupby("age")
    .apply(lambda x: x.sample(2))
)

df.to_csv("imdb_age_sampled.csv", index=False)
