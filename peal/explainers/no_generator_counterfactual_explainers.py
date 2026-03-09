import os
import torch
import pandas as pd
import numpy as np
from typing import Union

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dependencies", "DiCE")))
try:
    import dice_ml
except ImportError as e:
    print(f"Warning: Could not import dice_ml: {e}")
from peal.training.interfaces import PredictorConfig
from peal.explainers.interfaces import ExplainerConfig, ExplainerInterface

class DiCEExplainerConfig(ExplainerConfig):
    explainer_type: str = "DiCEExplainer"
    predictor_path: Union[str, type(None)] = None
    distilled_predictor: Union[type(None), str, dict] = None
    y_target_goal_confidence: float = 0.51
    num_attempts: int = 1
    parallel_attempts: int = 1
    # DiCE specific params
    method: str = "random" # "random", "genetic", "kdtree"
    features_to_vary: Union[list, str] = "all"
    proximity_weight: float = 0.5
    sparsity_weight: float = 0.5
    diversity_weight: float = 0.5


class DiCEExplainer(ExplainerInterface):
    """
    Explainer wrapper for DiCE (Diverse Counterfactual Explanations)
    https://arxiv.org/pdf/1905.07697
    """

    def __init__(self, config=None, explainer_config=None, predictor=None, predictor_datasources=None, datasource=None, **kwargs):
        self.explainer_config = explainer_config if explainer_config is not None else config
        self.predictor = predictor
        self.predictor_datasources = datasource if datasource is not None else predictor_datasources
        
        self.device = "cuda" if next(self.predictor.parameters()).is_cuda else "cpu"
        self.counterfactuals_per_second = None
        
        if self.predictor_datasources is not None and len(self.predictor_datasources) > 0:
            if hasattr(self.predictor_datasources[0], "dataset"):
                self.dataset = self.predictor_datasources[0].dataset
            else:
                self.dataset = self.predictor_datasources[0]
                
            self.setup_dice()
        else:
            self.dataset = None

    def setup_dice(self):
        # We need the full dataframe to initialize dice_ml.Data
        # Assuming self.dataset has underlying dataframe and attributes
        if hasattr(self.dataset, 'df') and hasattr(self.dataset, 'attributes'):
            df = self.dataset.df
            features = self.dataset.attributes
            # By default assume the target is the last attribute if not explicitly specified
            if hasattr(self.dataset, 'task_config') and self.dataset.task_config is not None and self.dataset.task_config.y_selection:
                outcome_name = self.dataset.task_config.y_selection[0]
            else:
                outcome_name = features[-1]
            
            # Select only the features used by the model
            if hasattr(self.dataset, 'task_config') and self.dataset.task_config is not None and self.dataset.task_config.x_selection:
                self.feat_names = self.dataset.task_config.x_selection
                df = df[self.feat_names + [outcome_name]]
            else:
                self.feat_names = [f for f in features if f != outcome_name]
                df = df[self.feat_names + [outcome_name]]
            # Find continuous features
            continuous_features = []
            if hasattr(self.dataset, 'continuous_features') and getattr(self.dataset, 'continuous_features'):
                continuous_features = self.dataset.continuous_features
            else:
                # heuristic: if dtype is float/int and many unique values
                for col in df.columns:
                    if col != outcome_name and pd.api.types.is_numeric_dtype(df[col]):
                        if len(df[col].unique()) > 10:
                            continuous_features.append(col)
            # 1. Init Data
            self.dice_data = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name=outcome_name)

            # 2. Init Model
            # We need to wrap the PyTorch predictor
            class PytorchWrapper:
                def __init__(self, predictor, device):
                    self.predictor = predictor
                    self.device = device
                def predict(self, X):
                    # X is a numpy array or dataframe
                    if isinstance(X, pd.DataFrame):
                        X = X.values
                    X = X.astype(np.float32)
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    # We assume multiclass/binary classification returning logits or probs
                    # Return class predictions
                    with torch.no_grad():
                        preds = torch.argmax(self.predictor(X_tensor), dim=-1).cpu().numpy()
                    return preds
                    
                def predict_proba(self, X):
                    if isinstance(X, pd.DataFrame):
                        X = X.values
                    X = X.astype(np.float32)
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        logits = self.predictor(X_tensor)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    return probs

            self.dice_model = dice_ml.Model(model=PytorchWrapper(self.predictor, self.device), backend="sklearn")

            # 3. Init Dice
            self.dice = dice_ml.Dice(self.dice_data, self.dice_model, method=self.explainer_config.method)
        else:
            print("Warning: DiCEExplainer requires the dataset to have a `df` attribute to initialize `dice_ml.Data`.")

    def explain_batch(self, batch, **kwargs):
        """
        batch is a dictionary containing at least:
        - "x_list": the input factuals (list of tensors or tensor)
        - "y_target_list": the target classes
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided in predictor_datasources to use DiCEExplainer.")
            
        x_in = batch["x_list"]
        if isinstance(x_in, list):
            x_in = torch.stack(x_in)
            
        y_targets = batch["y_target_list"]
        
        # Calculate source predictions and start confidences
        with torch.no_grad():
            logits_orig = self.predictor(x_in.to(self.device))
            probs_orig = torch.softmax(logits_orig, dim=-1)
            y_sources = torch.argmax(probs_orig, dim=-1)
            y_source_list = [y.item() for y in y_sources]
            y_target_start_confidence_list = [probs_orig[j, int(y_targets[j])].item() for j in range(len(x_in))]

        # Convert inputs back to dataframe for DiCE
        # This requires the dataset to know how to map tensor to original features
        feat_names = self.feat_names
        
        x_counterfactual_list = []
        y_target_end_confidence_list = []
        
        # New lists to return
        x_list_new = []
        y_list_new = []
        y_target_list_new = []
        y_source_list_new = []
        y_target_start_confidence_list_new = []
        
        # DiCE explains instances one by one or in blocks, but usually expects a dataframe
        for i in range(len(x_in)):
            x_single = x_in[i].detach().cpu().numpy()
            y_target = int(y_targets[i] if isinstance(y_targets, (list, tuple)) else y_targets[i].item())
            
            # Create a dataframe for the single query instance
            query_instance = pd.DataFrame([x_single], columns=feat_names)
            
            # CRITICAL FIX: Cast categorical columns back to int/object
            # PyTorch tensors are float32 (e.g., 0.0, 1.0). If DiCE's internal dataset has integers,
            # it will reject '1.0' as "outside the dataset".
            # CRITICAL FIX: Categorical Alignment for DiCE
            # PyTorch models predict via float32 tensors (e.g. 1.0) but DiCE internally maps
            # categories directly from the original DataFrame types (e.g. int `1` or object `'1'`).
            # Direct astype() often fails to match DiCE's internal unique value constraints.
            # Here we snap the float value to the closest explicit true categorical value
            # found in the training data distribution.
            for col in query_instance.columns:
                if col not in self.dice_data.continuous_feature_names:
                    query_val = float(query_instance[col].values[0])
                    # Get exact valid values DiCE extracted during setup
                    valid_values = self.dice_data.data_df[col].unique()
                    
                    # Find closest exact match (this handles int, string, object mappings universally)
                    # by checking numerical equivalence of valid_values
                    try:
                        valid_floats = np.array([float(x) for x in valid_values])
                        closest_idx = np.argmin(np.abs(valid_floats - query_val))
                        query_instance[col] = valid_values[closest_idx]
                    except ValueError:
                        # Fallback if categories are pure strings not numerically parsable
                        pass

                    # Explicitly convert the column to object to prevent DiCE's internal 
                    # Scikit-learn LabelEncoder from attempting continuous memory bindings 
                    # on large numeric indices.
                    query_instance[col] = query_instance[col].astype(object)
            
            # Debug traces
            # print("--- DiCE Query Instance ---")
            # print(query_instance)
            # print(query_instance.dtypes)
            # print("--- DiCE Expected Dtypes ---")
            # print(self.dice_data.data_df.drop(columns=[self.dice_data.outcome_name]).dtypes)
            
            # Generate counterfactuals
            found_cfs = []
            found_confs = []
            try:
                dice_exp = self.dice.generate_counterfactuals(
                    query_instance, 
                    total_CFs=self.explainer_config.num_attempts, 
                    desired_class=y_target,
                    features_to_vary=self.explainer_config.features_to_vary
                )
                
                cfs_df = dice_exp.cf_examples_list[0].final_cfs_df
                if cfs_df is not None and len(cfs_df) > 0:
                    for _, row in cfs_df.iterrows():
                        # Crucial Fix: DiCE reorders columns internally (e.g. continuous features last)
                        # We must explicitly slice the row using the exact original feat_names order
                        cf_features = row[self.feat_names].values.astype(np.float32)
                        cf_tensor = torch.tensor(cf_features, dtype=torch.float32).to(x_in.device)
                        with torch.no_grad():
                            logits = self.predictor(cf_tensor.unsqueeze(0).to(self.device))
                            conf = torch.softmax(logits, dim=-1)[0, y_target].item()
                        found_cfs.append(cf_tensor)
                        found_confs.append(conf)
                
                if not found_cfs:
                    found_cfs = [x_in[i]]
                    found_confs = [0.0]
                    
                if self.explainer_config.merge_clusters != "concatenate":
                    best_idx = np.argmax(found_confs)
                    found_cfs = [found_cfs[best_idx]]
                    found_confs = [found_confs[best_idx]]
                    
            except Exception as e:
                print(f"DiCE failed to generate CF: {e}")
                found_cfs = [x_in[i]]
                found_confs = [0.0]
            
            for cf, conf in zip(found_cfs, found_confs):
                x_list_new.append(x_in[i])
                y_list_new.append(batch["y_list"][i] if "y_list" in batch else None)
                y_target_list_new.append(y_targets[i])
                y_source_list_new.append(y_source_list[i])
                y_target_start_confidence_list_new.append(y_target_start_confidence_list[i])
                x_counterfactual_list.append(cf)
                y_target_end_confidence_list.append(conf)

        batch["x_list"] = x_list_new
        if "y_list" in batch: batch["y_list"] = y_list_new
        batch["y_target_list"] = y_target_list_new
        batch["y_source_list"] = y_source_list_new
        batch["y_target_start_confidence_list"] = y_target_start_confidence_list_new
        batch["x_counterfactual_list"] = x_counterfactual_list
        batch["y_target_end_confidence_list"] = y_target_end_confidence_list
        
        batch["z_difference_list"] = [None] * len(x_counterfactual_list)
        batch["history_list"] = []
        batch["cluster_list"] = []

        if self.explainer_config.tracking_level >= 4:
            base_path = kwargs.get("base_path", "collages")
            start_idx = kwargs.get("start_idx", 0)
            
            (
                batch["x_attribution_list"],
                batch["collage_path_list"],
            ) = self.dataset.generate_contrastive_collage(
                base_path=base_path,
                start_idx=start_idx,
                **batch
            )
        else:
            batch["x_attribution_list"] = [torch.abs(cf - orig) for cf, orig in zip(x_counterfactual_list, x_list_new)]
            batch["collage_path_list"] = [None] * len(x_counterfactual_list)

        return batch

    def cluster_explanations(self, *args, **kwargs):
        # Dummy implementation since tabular DiCE explanations do not require clustering
        return args[0] if args else None
