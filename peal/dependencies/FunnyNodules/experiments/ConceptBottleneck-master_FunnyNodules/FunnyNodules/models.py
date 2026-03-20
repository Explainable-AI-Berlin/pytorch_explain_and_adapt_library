
import torch
import torch.nn as nn
import numpy as np
from FunnyNodules import analysis
from torch.autograd import Variable
from sklearn.svm import SVR
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from OAI.template_model import PretrainedResNetModel
from OAI.intervention_model import InterventionModelOnC

class ModelXtoCY(PretrainedResNetModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.C_cols = cfg['C_cols']
        self.y_cols = cfg['y_cols']

        # FC layers to use for outputs
        self.C_fc_name = cfg['C_fc_name']
        self.y_fc_name = cfg['y_fc_name']

        self.C_loss_type = cfg['C_loss_type']
        self.y_loss_type = cfg['y_loss_type']
        self.classes_per_C_col = cfg['classes_per_C_col']
        self.classes_per_y_col = cfg['classes_per_y_col']

        self.C_loss_weigh_class = cfg['C_loss_weigh_class']
        self.additional_loss_weighting = cfg['additional_loss_weighting']
        self.metric_to_use_as_stopping_criterion = 'val_mal_negative_rmse'#'val_xrkl_negative_rmse'

    def get_data_dict_from_dataloader(self, data):

        X = data[0].repeat(1, 3, 1, 1)#data['image']  # X
        y = data[3]#data['y']  # y
        C_feats = data[2]#data['C_feats']
        ones_array = torch.from_numpy(np.ones(data[2].shape)).float().cuda()#sanity check np.zeros -> attris schlechtrandom
        C_feats_not_nan = data[-1]#ones_array#data['C_feats_not_nan']
        C_feats_loss_class_wts = ones_array#data['C_feats_loss_class_wts']

        # Wrap them in Variable
        X = Variable(X.float().cuda())
        y = Variable(y.float().cuda())
        if len(self.C_cols) > 0:
            C_feats = Variable(C_feats.float().cuda())
            C_feats_not_nan = Variable(C_feats_not_nan.float().cuda())

        inputs = { 'image': X }
        labels = { 'y': y,
                   'C_feats': C_feats,
                   'C_feats_not_nan': C_feats_not_nan,
                   'C_feats_loss_class_wts': C_feats_loss_class_wts }

        data_dict = {
            'inputs': inputs,
            'labels': labels,
        }
        return data_dict

    def forward(self, inputs):
        x = inputs['image']
        print("forward in modelxtocy")
        print(x.shape)
        x = self.compute_cnn_features(x)

        outputs = {} # { 'pool': x }
        N_layers = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.y_fc_name:
                assert i == N_layers - 1 # To ensure that we are at last layer
                assert fc_name == self.C_fc_name # C + y should be at the same fc layer
                N_y_cols = len(self.y_cols)
                N_C_cols = len(self.C_cols)
                # No ReLu
                outputs['y'] = x[:, :N_y_cols]
                outputs['C'] = x[:, N_y_cols:N_y_cols + N_C_cols]
                continue

            x = self.relu(x)
            # Can choose to keep track of this as well
            # outputs[fc_name] = x

        return outputs

    def loss(self, outputs, data_dict):
        # Data
        y = data_dict['labels']['y']
        C_feats = data_dict['labels']['C_feats']
        C_feats_not_nan = data_dict['labels']['C_feats_not_nan']
        C_feats_loss_class_wts = data_dict['labels']['C_feats_loss_class_wts']

        # Parse outputs from deep model
        C_hat = outputs['C']
        y_hat = outputs['y']

        # Loss for y
        if self.y_loss_type == 'reg':
            loss_y = nn.MSELoss()(input=y_hat, target=y)
        elif self.y_loss_type == 'cls':
            y_long = y.long()
            loss_y = nn.CrossEntropyLoss()(y_hat, y_long)
            # loss_y = []
            # start_id = 0
            # print(y_hat.shape)
            # print(y_long.shape)
            # for i, N_cls in enumerate(self.classes_per_y_col):
            #     loss_y.append(nn.CrossEntropyLoss(reduction='none')(y_hat[:, start_id:start_id + N_cls],
            #                                                         y_long[:, i]))
            #     start_id += N_cls
            # loss_y = torch.stack(loss_y, dim=1)
            # loss_y = loss_y.sum(dim=1).mean(dim=0)
            # assert start_id == sum(self.classes_per_y_col)

        # Loss for C
        if self.C_loss_type == 'reg':
            # loss_C = nn.MSELoss()(input=C_hat, target=C_feats)
            loss_C = ((C_feats - C_hat) ** 2)
        elif self.C_loss_type == 'cls':
            C_feats_long = C_feats.long()
            loss_C = []
            start_id = 0
            for i, N_cls in enumerate(self.classes_per_C_col):
                loss_C.append(nn.CrossEntropyLoss(reduction='none')(C_hat[:, start_id:start_id + N_cls],
                                                                    C_feats_long[:, i]))
                start_id += N_cls
            loss_C = torch.stack(loss_C, dim=1)
            assert start_id == sum(self.classes_per_C_col)

        # Compute loss only if feature is not NaN.
        loss_C = loss_C * C_feats_not_nan
        # We upweigh rare classes (within each concept) to allow the model to pay attention to it.
        if self.C_loss_weigh_class:
            loss_class_wts = C_feats_loss_class_wts
            loss_C *= loss_class_wts.float().cuda()
        loss_C *= torch.FloatTensor([self.additional_loss_weighting]).cuda()
        loss_C = loss_C.sum(dim=1).mean(dim=0)

        # Final loss
        loss = loss_y + loss_C
        loss /= (sum(self.additional_loss_weighting) + 1.)

        loss_y_float = loss_y.data.cpu().numpy().flatten()
        loss_C_float = loss_C.data.cpu().numpy().flatten()
        loss_ratio = loss_y_float / loss_C_float

        # Use y only and no C
        # loss = nn.MSELoss()(input=y_hat, target=y)

        loss_details = {
            'loss_ratio': loss_ratio
        }
        return loss, loss_details

    def analyse_predictions(self, y_true, y_pred, info={}):
        # This function is called at the end of every train / val epoch in DeepLearningModel and is
        # used to analyse the predictions over the entire train / val dataset.
        metrics_all = {}
        phase = info['phase']
        dataset_size = info['dataset_size']
        epoch_loss = info['epoch_loss']
        loss_ratios = np.concatenate([x['loss_ratio'] for x in info['loss_details']])

        print('Loss_y divided by loss_C is %2.3f (median ratio across batches)' % np.median(loss_ratios))

        y_hat = y_pred['y']
        C_hat = y_pred['C']
        y = y_true['y']
        C = y_true['C_feats']
        assert len(y_hat) == dataset_size
        # print("y_hat"+str(y_hat.shape))
        # print("y"+str(y.shape))
        # print("C_hat"+str(C_hat.shape))
        # print("C"+str(C.shape))

        if self.y_loss_type == 'cls':
            y_hat_new = []
            start_id = 0
            for i, N_cls in enumerate(self.classes_per_y_col):
                y_hat_new.append(np.argmax(y_hat[:, start_id:start_id + N_cls], axis=1))
                start_id += N_cls
            y_hat = np.stack(y_hat_new, axis=1).astype(np.float32)
            assert start_id == sum(self.classes_per_y_col)

        y = np.expand_dims(y, axis=1)

        if self.C_loss_type == 'cls':
            C_hat_new = []
            start_id = 0
            for i, N_cls in enumerate(self.classes_per_C_col):
                C_hat_new.append(np.argmax(C_hat[:, start_id:start_id + N_cls], axis=1))
                start_id += N_cls
            C_hat = np.stack(C_hat_new, axis=1).astype(np.float32)
            assert start_id == sum(self.classes_per_C_col)

        metrics_y = analysis.assess_performance(y=y, yhat=y_hat,
                                                names=self.y_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix=phase,
                                                verbose=True)

        metrics_C = analysis.assess_performance(y=C, yhat=C_hat,
                                                names=self.C_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix=phase)
        # print("all metrics")
        # print(metrics_y)
        # print(metrics_C)
        if phase == 'val':
            print("val_mal_acc "+str(metrics_y['val_mal_acc']))
            print("val_roundness_acc "+str(metrics_C['val_roundness_acc']))
            print("val_spiculation_acc "+str(metrics_C['val_spiculation_acc']))
            print("val_edge_sharpness_acc "+str(metrics_C['val_edge_sharpness_acc']))
            print("val_size_attr_acc "+str(metrics_C['val_size_attr_acc']))
            print("val_intensity_acc "+str(metrics_C['val_intensity_acc']))
            print("val_internal_structure_acc "+str(metrics_C['val_internal_structure_acc']))
        print('%s epoch loss for %s: %2.6f; RMSE %2.6f; correlation %2.6f (n=%i)' %
              (phase, str(self.y_cols), epoch_loss, metrics_y['%s_mal_rmse' % phase], metrics_y['%s_mal_r' % phase],
               len(y_hat)))
        metrics_all['%s_epoch_loss' % phase] = epoch_loss

        rmses = [metrics_C['%s_%s_rmse' % (phase, C_col)] for C_col in self.C_cols]
        print('  Average RMSE for C: %2.4f, RMSEs: %s' % (np.mean(rmses), str(rmses)))

        metrics_all.update(metrics_y)
        metrics_all.update(metrics_C)
        return metrics_all

    def analyse_FunnyNodules(self):
        print("do test_contrastivity")
        self.test_contrastivity()
        print("do visualize_all_attribute_responses")
        self.visualize_all_attribute_responses()
        print("do visualize_roundness_cs_interaction")
        self.visualize_roundness_cs_interaction()

    def test_contrastivity(self):
        import pandas as pd
        import random
        from matplotlib import pyplot as plt
        from FunnyNodules.dataset import create_attribute_dataloader
        attris = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure"]
        results = []
        for s in list(range(0, 100)):# [random.randint(1, 1000) for _ in range(100)]:
            for attris_idx, attr_name in enumerate(attris):
                dl = create_attribute_dataloader(attr_name, base_seed=s)
                all_y_hat = []
                all_C_hat = []
                all_y = []
                all_C = []
                with torch.no_grad():
                    for i, data in enumerate(dl):
                        (x,y_mask, y_attributes, y_mal, sampleID, _, _) = data
                        data_dict = self.get_data_dict_from_dataloader(data)
                        inputs = data_dict['inputs']
                        labels = data_dict['labels']
                        outputs = self.forward(inputs)

                        batch_y_hat = outputs['y']
                        batch_C_hat = outputs['C']
                        batch_y = labels['y']
                        batch_C = labels['C_feats']

                        all_y_hat.append(batch_y_hat.detach().cpu().numpy())
                        all_C_hat.append(batch_C_hat.detach().cpu().numpy())
                        all_y.append(batch_y.detach().cpu().numpy())
                        all_C.append(batch_C.detach().cpu().numpy())

                all_y_hat = np.concatenate(all_y_hat, axis=0)
                all_C_hat = np.concatenate(all_C_hat, axis=0)
                all_y = np.concatenate(all_y, axis=0)
                all_C = np.concatenate(all_C, axis=0)
                if self.y_loss_type == 'cls':
                    y_hat_new = []
                    start_id = 0
                    for i, N_cls in enumerate(self.classes_per_y_col):
                        y_hat_new.append(np.argmax(all_y_hat[:, start_id:start_id + N_cls], axis=1))
                        start_id += N_cls
                    all_y_hat = np.stack(y_hat_new, axis=1).astype(np.float32)
                    assert start_id == sum(self.classes_per_y_col)
                all_y = np.expand_dims(all_y, axis=1)
                if self.C_loss_type == 'cls':
                    C_hat_new = []
                    start_id = 0
                    for i, N_cls in enumerate(self.classes_per_C_col):
                        C_hat_new.append(np.argmax(all_C_hat[:, start_id:start_id + N_cls], axis=1))
                        start_id += N_cls
                    all_C_hat = np.stack(C_hat_new, axis=1).astype(np.float32)
                    assert start_id == sum(self.classes_per_C_col)

                metrics = {}
                if all_y.shape[1] == 1:
                    y_int = np.round(all_y[:, 0]).astype(int)
                    yhat_diff = np.abs(all_y_hat[:, 0] - y_int)
                    within_1 = np.mean(yhat_diff <= 1)
                    metrics['test_contrastivity' + '_' + 'y' + '_within_1_score'] = within_1
                else:
                    raise ValueError("y has wrong shape")
                if all_C.shape[1] == 6:
                    final_ranges = [4, 4, 4, 4, 4, 1]
                    for i, (name, max_val) in enumerate(zip(attris, final_ranges)):
                        C_scaled = all_C[:, i] * max_val
                        Chat_scaled = all_C_hat[:, i] * max_val
                        C_rounded = np.round(C_scaled).astype(int)
                        Chat_rounded = np.round(Chat_scaled).astype(int)
                        if max_val == 1:
                            within_1 = np.mean(np.abs(Chat_rounded - C_rounded) == 0)
                        else:
                            within_1 = np.mean(np.abs(Chat_rounded - C_rounded) <= 1)
                        metric_key = f"{'test_contrastivity'}_{name}_within_1_score"
                        metrics[metric_key] = within_1
                else:
                    raise ValueError("C has wrong shape")

                entry = {
                    "seed": s,
                    "varied_attribute": attr_name,
                    "test_acc_contra": float(metrics['test_contrastivity_y_within_1_score']),
                }
                for i, attr in enumerate(attris):
                    entry[f"attr_acc_{attr}"] = float(metrics['test_contrastivity_'+attr+'_within_1_score'])
                results.append(entry)
        results_df = pd.DataFrame(results)
        summary = results_df.groupby("varied_attribute")[
            ["test_acc_contra",
             "attr_acc_roundness", "attr_acc_spiculation", "attr_acc_edge_sharpness",
             "attr_acc_size_attr", "attr_acc_intensity", "attr_acc_internal_structure"]
        ].agg(["mean", "std"])
        pd.set_option('display.max_columns', None)  # show all columns
        pd.set_option('display.width', 0)  # don't wrap lines
        pd.set_option('display.max_colwidth', None)  # show full column names
        print(summary.round(3))
        # === Plot model test accuracy by varied attribute ===
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Left plot: test accuracy per varied attribute
        results_df.boxplot(column="test_acc_contra", by="varied_attribute", ax=axs[0])
        axs[0].set_title("Model Test Accuracy by Varied Attribute")
        axs[0].set_xlabel("Varied Attribute")
        axs[0].set_ylabel("Accuracy")
        # Right plot: heatmap-style barplot for attribute accuracies
        attr_cols = [
            "attr_acc_roundness", "attr_acc_spiculation", "attr_acc_edge_sharpness",
            "attr_acc_size_attr", "attr_acc_intensity", "attr_acc_internal_structure"
        ]
        # Compute mean accuracies for each varied_attribute
        attr_summary = results_df.groupby("varied_attribute")[attr_cols].mean()
        attr_summary.plot(kind="bar", ax=axs[1])
        axs[1].set_title("Attribute Prediction Accuracies (mean per varied attribute)")
        axs[1].set_xlabel("Varied Attribute")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend(title="Predicted Attribute", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.suptitle("")
        plt.tight_layout()
        #plt.show()

    def visualize_all_attribute_responses(self):
        from matplotlib import pyplot as plt
        from FunnyNodules.dataset import create_attribute_dataloader
        attributes = [
            "roundness", "spiculation", "edge_sharpness",
            "size_attr", "intensity", "internal_structure"
        ]
        seeds = list(range(0, 100)) #[random.randint(1, 1000) for _ in range(100)]
        self.eval()
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs = axs.flatten()
        for i, attr in enumerate(attributes):
            all_varied = []
            all_true = []
            all_pred = []
            for s in seeds:
                dl = create_attribute_dataloader(attr, base_seed=s)
                varied_values = []
                true_targets = []
                pred_targets = []
                with torch.no_grad():
                    for _, data in enumerate(dl):
                        (x,y_mask, y_attributes, y_mal, sampleID, _, _) = data
                        data_dict = self.get_data_dict_from_dataloader(data)
                        inputs = data_dict['inputs']
                        labels = data_dict['labels']
                        outputs = self.forward(inputs)
                        preds = outputs['y'].detach().cpu()
                        targets = labels['y'].detach().cpu().numpy()

                        if self.y_loss_type == 'cls':
                            y_hat_new = []
                            start_id = 0
                            for _, N_cls in enumerate(self.classes_per_y_col):
                                y_hat_new.append(np.argmax(preds[:, start_id:start_id + N_cls], axis=1))
                                start_id += N_cls
                            preds = np.stack(y_hat_new, axis=1).astype(np.float32)
                            assert start_id == sum(self.classes_per_y_col)
                        targets = np.expand_dims(targets, axis=1)
                        pred_targets.extend((preds/4).tolist())
                        true_targets.extend((targets/4).tolist())
                        varied_values.extend(
                            labels['C_feats'][:, attributes.index(attr)].cpu().numpy().tolist()
                        )
                order = np.argsort(varied_values)
                all_varied.append(np.array(varied_values)[order])
                all_true.append(np.array(true_targets)[order])
                all_pred.append(np.array(pred_targets)[order])

            all_varied = np.array([v.flatten() for v in all_varied])
            all_true = np.array([v.flatten() for v in all_true])
            all_pred = np.array([v.flatten() for v in all_pred])
            mean_true = all_true.mean(axis=0)
            mean_pred = all_pred.mean(axis=0)
            std_pred = all_pred.std(axis=0)
            x_vals = all_varied[0]
            print('visualize_all_attribute_responses')
            print(attr)
            print('x_vals '+str(x_vals))
            print('mean_true '+str(mean_true))
            print('mean_pred '+str(mean_pred))
            # Plot
            ax = axs[i]
            ax.plot(x_vals, mean_true, 'o-', color='black', label='Ground Truth')
            ax.plot(x_vals, mean_pred, 's--', color='tab:blue', label='Predicted')
            ax.fill_between(x_vals, mean_pred - std_pred, mean_pred + std_pred, color='tab:blue', alpha=0.2)
            ax.set_title(attr)
            ax.set_xlabel("Attribute Value (normalized 0–1)")
            ax.set_ylabel("Target Value (normalized 0→1)")
            ax.grid(True)
            if i == 0:
                ax.legend()

        plt.suptitle("Model Target Sensitivity", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        #plt.show()

    def visualize_roundness_cs_interaction(self):
        import torch
        from matplotlib import pyplot as plt
        from FunnyNodules.dataset import create_attribute_dataloader
        self.eval()
        roundness_values = [1, 2, 3, 4, 5]
        seeds = list(range(0, 100)) #[random.randint(1, 1000) for _ in range(100)]
        # store results separated by IS
        results = {0: {"true": [], "pred": []}, 1: {"true": [], "pred": []}}
        for s in seeds:
            dl = create_attribute_dataloader(
                attribute_name="roundness",
                base_seed=s,
                img_size=224,
                batch_size=1,
                fix_all_but_r_and_is=True
            )
            true_targets_is = {0: [], 1: []}
            pred_targets_is = {0: [], 1: []}

            with torch.no_grad():
                for _, data in enumerate(dl):
                    (x, y_mask, y_attributes, y_mal, sampleID, _, _) = data
                    data_dict = self.get_data_dict_from_dataloader(data)
                    inputs = data_dict['inputs']
                    labels = data_dict['labels']
                    outputs = self.forward(inputs)
                    preds = outputs['y'].detach().cpu()
                    targets = labels['y'].detach().cpu().numpy()
                    if self.y_loss_type == 'cls':
                        y_hat_new = []
                        start_id = 0
                        for _, N_cls in enumerate(self.classes_per_y_col):
                            y_hat_new.append(np.argmax(preds[:, start_id:start_id + N_cls], axis=1))
                            start_id += N_cls
                        preds = np.stack(y_hat_new, axis=1).astype(np.float32).squeeze()
                        assert start_id == sum(self.classes_per_y_col)

                    is_val = labels['C_feats'][0,-1].item()

                    pred_targets_is[is_val].append((preds.item()/4))
                    true_targets_is[is_val].append((targets.item()/4))


            # append to results
            for cs in [0, 1]:
                if len(true_targets_is[cs]) > 0:
                    results[cs]["true"].append(true_targets_is[cs])
                    results[cs]["pred"].append(pred_targets_is[cs])
        # Convert to arrays and compute mean/std across seeds
        plt.figure(figsize=(7, 5))
        for cs in [0, 1]:
            all_true = np.array(results[cs]["true"])
            all_pred = np.array(results[cs]["pred"])
            mean_true = all_true.mean(axis=0)
            mean_pred = all_pred.mean(axis=0)
            std_pred = all_pred.std(axis=0)
            color_gt = 'black' if cs == 0 else 'gray'
            color_pred = 'tab:blue' if cs == 0 else 'tab:orange'
            plt.plot(roundness_values, mean_true, 'o-', color=color_gt, label=f"GT is={cs}")
            plt.plot(roundness_values, mean_pred, 's--', color=color_pred, label=f"Pred is={cs}")
            plt.fill_between(roundness_values, mean_pred - std_pred, mean_pred + std_pred,
                             color=color_pred, alpha=0.2)
            print('visualize_roundness_is_interaction')
            print('is '+str(cs))
            print('roundness_values '+str(roundness_values))
            print('mean_true '+str(mean_true))
            print('mean_pred '+str(mean_pred))

        plt.xlabel("Roundness")
        plt.ylabel("Target Value (normalized 0→1)")
        plt.title("Effect of Roundness on Target conditioned on IS with other attributes=3")
        plt.grid(True)
        plt.legend()
        #plt.show()

class ModelXtoCtoY(InterventionModelOnC, ModelXtoCY):
    def __init__(self, cfg):
        InterventionModelOnC.__init__(self, cfg)
        ModelXtoCY.__init__(self, cfg)

    def forward(self, inputs):
        x = inputs['image']
        x = self.compute_cnn_features(x)
        x = self.dropout(x)

        outputs = {}  # { 'pool': x }
        N_layers = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.C_fc_name:
                # No ReLu for concept layer
                outputs['C'] = x
                continue
            elif fc_name == self.y_fc_name:
                assert i == N_layers - 1
                # No ReLu for y layer
                outputs['y'] = x
                continue

            x = self.relu(x)
            # x = self.dropout(x)
            # Can choose to keep track of this as well
            # outputs[fc_name] = x

        # print("outputs")
        # print(outputs['C'].shape)
        # print(outputs['y'].shape)
        # raise ValueError('test')
        return outputs