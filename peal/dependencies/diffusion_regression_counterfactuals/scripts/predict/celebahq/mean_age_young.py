import json
import pandas as pd

# Define the path to the CSV file
csv_file_path = (
    "/home/space/datasets/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
)

# Load the CSV file, skipping the first two lines and without a header
df = pd.read_csv(csv_file_path, skiprows=2, header=None, sep=r" +", engine="python")

# Assign column names
columns = ["filename"] + [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]
df.columns = columns


df_age = pd.read_csv("../CelebAMask-HQ.csv")
df_age["filename"] = df_age["file"].apply(lambda x: x.split("/")[-1])
df_age.rename(columns={"prediction": "age"}, inplace=True)

# Merge the two dataframes on the "filename" column
df_merged = pd.merge(df, df_age, on="filename")
df_young = df_merged[df_merged["Young"] == 1]
df_old = df_merged[df_merged["Young"] != 1]

# Calculate some interesting statistics for the "Young" and "Old" groups
count_young = df_young.shape[0]
count_old = df_old.shape[0]

mean_age_young = df_young["age"].mean()
std_age_young = df_young["age"].std()
mean_age_old = df_old["age"].mean()
std_age_old = df_old["age"].std()

# Save to json
result = {
    "count_young": count_young,
    "count_old": count_old,
    "mean_age_young": mean_age_young,
    "mean_age_old": mean_age_old,
    "std_age_young": std_age_young,
    "std_age_old": std_age_old,
}

json.dump(result, open("mean_age_young.json", "w"), indent=2)
