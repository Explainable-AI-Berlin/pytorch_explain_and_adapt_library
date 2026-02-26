import os
import torch.nn.functional as F

from peal.data.datasets import SymbolicDataset

def setup_kaggle_api():
    """Helper to ensure Kaggle API is authenticated, prompting the user if necessary."""
    import os
    import json
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json_path):
        os.makedirs(kaggle_dir, exist_ok=True)
        print("Kaggle API credentials not found.")
        username = input("Please enter your Kaggle username: ").strip()
        key = input("Please enter your Kaggle API key: ").strip()
        
        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)
        print(f"Credentials saved to {kaggle_json_path}.")
        
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

from peal.data.dataset_generators import CircleDatasetGenerator

class CircleDataset(SymbolicDataset):
    def __init__(self, mode, config, **kwargs):
        if not os.path.exists(config.dataset_path):
            # use the circle dataset generator
            circle_dataset_generator = CircleDatasetGenerator(config)
            circle_dataset_generator.generate_dataset()
        super(CircleDataset, self).__init__(mode=mode, config=config, **kwargs)
        self.hints_enabled = False
        self.idx_enabled = False

    def calculate_outlier_score(self, x):
        import torch
        outlier_scores = {"absolute" : torch.zeros(x.shape[0], device=x.device)}
        if hasattr(self, "reference_outlier_scores") and self.reference_outlier_scores is not None:
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores

class AdultDataset(SymbolicDataset):
    def __init__(self, mode, config, **kwargs):
        if not os.path.exists(f"{config.dataset_path}/data.csv"):
            print(f"Dataset path {config.dataset_path} not found. Attempting to download via Kaggle API...")
            os.makedirs(config.dataset_path, exist_ok=True)
            try:
                api = setup_kaggle_api()
                api.dataset_download_files('wenruliu/adult-income-dataset', path=config.dataset_path, unzip=True)
                # Ensure the downloaded file is processed for SymbolicDataset conformity
                import pandas as pd
                csv_file = f"{config.dataset_path}/adult.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    # Handle target col
                    if 'income' in df.columns:
                        df['Target'] = df.pop('income').astype('category').cat.codes
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype('category').cat.codes
                    df = df.astype(float)
                    df.to_csv(f"{config.dataset_path}/data.csv", index=False)
                    os.remove(csv_file)
            except Exception as e:
                print(f"Failed to download Adult dataset automatically: {e}")
                print(f"Please manually download the Adult dataset from Kaggle and place it as: {config.dataset_path}/data.csv")
                import time; time.sleep(5)
            
        with open(f"{config.dataset_path}/data.csv", "r") as f:
            config.input_size = [len(f.readline().strip().split(",")) - 1]
        super(AdultDataset, self).__init__(mode, config, **kwargs)
        self.hints_enabled = False
        self.idx_enabled = False

    def calculate_outlier_score(self, x):
        import torch
        outlier_scores = {"absolute" : torch.zeros(x.shape[0], device=x.device)}
        if hasattr(self, "reference_outlier_scores") and self.reference_outlier_scores is not None:
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores

class CompassDataset(SymbolicDataset):
    def __init__(self, mode, config, **kwargs):
        if not os.path.exists(f"{config.dataset_path}/data.csv"):
            print(f"Dataset path {config.dataset_path} not found. Attempting to download COMPASS via Kaggle API...")
            os.makedirs(config.dataset_path, exist_ok=True)
            try:
                import pandas as pd
                print("Attempting to fetch COMPASS from FairML GitHub...")
                url = "https://raw.githubusercontent.com/DataResponsibly/fairDAGs/master/data/compas/propublica_data_for_fairml.csv"
                df = pd.read_csv(url)
                if 'Two_yr_Recidivism' in df.columns:
                    df['Target'] = df.pop('Two_yr_Recidivism').astype(float)
                # Ensure is_black is available for the teacher
                if 'African_American' in df.columns:
                    df['is_black'] = df['African_American'].astype(float)
                
                for col in df.columns:
                    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                        df[col] = df[col].astype('category').cat.codes
                df = df.astype(float)
                df.to_csv(f"{config.dataset_path}/data.csv", index=False)
            except Exception as e:
                print(f"Failed to download COMPASS dataset automatically: {e}")
                print(f"Please manually place the COMPASS dataset as: {config.dataset_path}/data.csv")
            
        with open(f"{config.dataset_path}/data.csv", "r") as f:
            config.input_size = [len(f.readline().strip().split(",")) - 1]
        super(CompassDataset, self).__init__(mode, config, **kwargs)
        self.hints_enabled = False
        self.idx_enabled = False

    def calculate_outlier_score(self, x):
        import torch
        outlier_scores = {"absolute" : torch.zeros(x.shape[0], device=x.device)}
        if hasattr(self, "reference_outlier_scores") and self.reference_outlier_scores is not None:
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores

class GermanDataset(SymbolicDataset):
    def __init__(self, mode, config, **kwargs):
        if not os.path.exists(f"{config.dataset_path}/data.csv"):
            print(f"Dataset path {config.dataset_path} not found. Attempting to download German Credit via Kaggle API...")
            os.makedirs(config.dataset_path, exist_ok=True)
            try:
                from sklearn.datasets import fetch_openml
                print("Attempting to fetch German Credit from OpenML...")
                data = fetch_openml(name='credit-g', version=1, as_frame=True)
                df = data.frame
                # Mapping target
                if 'class' in df.columns:
                    df['Target'] = df['class'].map({'good': 1, 'bad': 0}).astype(float)
                # Mapping confounder 'sex'
                if 'personal_status' in df.columns:
                     df['sex'] = df['personal_status'].apply(lambda x: 1.0 if 'male' in str(x).lower() else 0.0)
                
                for col in df.columns:
                    if df[col].dtype.name == 'category' or df[col].dtype == 'object':
                        df[col] = df[col].astype('category').cat.codes
                df = df.astype(float)
                df.to_csv(f"{config.dataset_path}/data.csv", index=False)
            except Exception as e:
                print(f"Failed to download German dataset automatically: {e}")
                print(f"Please manually place the German dataset as: {config.dataset_path}/data.csv")
                import time; time.sleep(5)
            
        with open(f"{config.dataset_path}/data.csv", "r") as f:
            config.input_size = [len(f.readline().strip().split(",")) - 1]
        super(GermanDataset, self).__init__(mode, config, **kwargs)
        self.hints_enabled = False
        self.idx_enabled = False

    def calculate_outlier_score(self, x):
        import torch
        outlier_scores = {"absolute" : torch.zeros(x.shape[0], device=x.device)}
        if hasattr(self, "reference_outlier_scores") and self.reference_outlier_scores is not None:
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores