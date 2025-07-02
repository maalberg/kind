import numpy as np
import torch
from matplotlib import pyplot as plt

import utils_data


def label_stationarity(dmd_model, dmd_residual_max, dataset_dir, data_timeseries_sz):
    """
    Based on DMD model residuals, creates a set of labels for stationary/non-stationary data.
    """
    data = utils_data.read_datafile(f'{dataset_dir}/test', data_timeseries_sz)
    labels = torch.zeros(data.shape[0], dtype=torch.bool)

    for this, timeseries in enumerate(data):
        timeseries = torch.unsqueeze(timeseries, 0)

        # --! predict timeseries
        o = dmd_model(timeseries, alpha=1.0)
        timeseries_predict = o[0]

        residual = torch.mean((timeseries - timeseries_predict)**2)
        label = True if residual < dmd_residual_max else False
        labels[this] = label

    return labels


def create_stationarity_dataset(dmd_model, dmd_residual_max, data_dirs, timeseries_nsample):
    """
    Returns a dataset which contains a list of tuples. The size of this list corresponds to the sum of
    stationary and non-stationary data as read from file folders ``data_dirs``. The tuple
    then consists of timeseries and a label, where label=1.0 denotes stationary
    timeseries, and label=0.0 non-stantionary.
    """
    labels_stationary    = label_stationarity(dmd_model, dmd_residual_max, data_dirs[0], timeseries_nsample)
    labels_nonstationary = label_stationarity(dmd_model, dmd_residual_max, data_dirs[1], timeseries_nsample)

    # --! convert boolean labels to floats
    labels_stationary    = labels_stationary.float()
    labels_nonstationary = labels_nonstationary.float()

    data_stationary      = read_datafile(f'{data_dirs[0]}/test', timeseries_nsample)
    data_nonstationary   = read_datafile(f'{data_dirs[1]}/test', timeseries_nsample)

    data   = torch.cat([data_stationary, data_nonstationary], dim=0)
    labels = torch.cat([labels_stationary, labels_nonstationary], dim=0)

    return torch.utils.data.TensorDataset(data, labels)


def train_alpha_fun(model, dataset, parameters):

    epochs_n = parameters['epochs_n']
    lr       = parameters['learning_rate']
    wd       = parameters['weight_decay']
    batsize  = parameters['batsize']

    # --! stratify data splits to ensure the two data classes are equally represented in both splits
    labels = dataset[:][1]
    indices_train, indices_valid, _, _ = train_test_split(range(len(dataset)), labels, stratify=labels, test_size=0.2)

    dataset_train = torch.utils.data.Subset(dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(dataset, indices_valid)

    # --! create data loaders for training and validation datasets
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batsize, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batsize, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCELoss()

    # --! run training loop
    for epoch in range(epochs_n):
        for xb, yb in dataloader_train:
            optimizer.zero_grad()

            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_preds = []
            val_targets = []
            for xb, yb in dataloader_valid:
                preds = model(xb).squeeze()
                val_preds.append(preds)
                val_targets.append(yb)
            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)

            # --! if prediction is greater than 0.5, it should equal True in targets,
            # --! and vice versa if it is less, is should correspond to False
            val_acc = ((val_preds > 0.5) == val_targets).float().mean()
            print(f"Epoch {epoch+1}, Val Acc: {val_acc:.3f}")

