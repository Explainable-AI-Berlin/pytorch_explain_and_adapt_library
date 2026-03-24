import pandas as pd

# get adult data from link
#df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None)
df = pd.read_csv("datasets/adult/data_original.csv")

print("Columns:")
print("\n".join(df.columns))


columns_to_take = ['hours-per-week', 'educational-num', 'occupation', 'workclass', 'race', 'age', 'marital-status', 'gender']
target_column = 'Target'

df = df[columns_to_take + [target_column]]

# transform target column to binary: 1 if income >50K, 0 otherwise
df[target_column] = df[target_column].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# save the curated dataset
#df.to_csv("datasets/adult/data.csv", index=False)

# output missing values
#print("Missing values:")
#print(df.isnull().sum())

# output head
print("Head of the dataset:")
print(df.head())

print("Missing values:")
print(df.isnull().sum())

# some preprocessing:
df['workclass'] = df['workclass'].replace({
    '?': 'Unknown',
    'Federal-gov': 'Government',
    'Local-gov': 'Government',
    'State-gov': 'Government',
    'Self-emp-inc': 'Self-Employed',
    'Self-emp-not-inc': 'Self-Employed',
    'Never-worked': 'Other',
    'Without-pay': 'Other',
})
df['workclass'] = df['workclass'].replace({'Other': 'Other/Unknown', 'Unknown': 'Other/Unknown'})
df['workclass'] = df['workclass'].astype('object')


df['occupation'] = df['occupation'].replace({
    '?': 'Other/Unknown',
    'Adm-clerical': 'White-Collar',
    'Craft-repair': 'Blue-Collar',
    'Exec-managerial': 'White-Collar',
    'Farming-fishing': 'Blue-Collar',
    'Handlers-cleaners': 'Blue-Collar',
    'Machine-op-inspct': 'Blue-Collar',
    'Other-service': 'Service',
    'Priv-house-serv': 'Service',
    'Prof-specialty': 'Professional',
    'Protective-serv': 'Service',
    'Tech-support': 'Service',
    'Transport-moving': 'Blue-Collar',
    'Unknown': 'Other/Unknown',
    'Armed-Forces': 'Other/Unknown',
})
df['occupation'] = df['occupation'].astype('object')


df['marital-status'] = df['marital-status'].replace({
    'Married-AF-spouse': 'Married',
    'Married-civ-spouse': 'Married',
    'Married-spouse-absent': 'Married',
    'Never-married': 'Single',
})
df['marital-status'] = df['marital-status'].astype('object')

print(df.shape)
print(df.info())

# print the distribution of each object-feature
print("Distribution of object features:")
object_columns = df.select_dtypes(include=['object']).columns
for col in object_columns:
    print(f"Feature: {col}")
    print(df[col].value_counts())
    print("\n")

# also print the distribution of the target variable
print("Distribution of target variable:")
print(df[target_column].value_counts())


# one-hot-encode the object features
df = pd.get_dummies(df, columns=object_columns, drop_first=True)

# apply normalization for numeric features
from sklearn.preprocessing import MinMaxScaler
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_column)
scaler = MinMaxScaler(feature_range=(-1, 1))
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

df.to_csv("datasets/adult/data.csv", index=False)