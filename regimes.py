# --!                                        --!
# --! nominal regime and excursion transient --!
# --!                                        --!

from abc import abstractmethod
from abc import ABC as interface
from collections import namedtuple

import numpy as np
import torch


regimes = namedtuple('regimes', 'nominal excursion')


class encoder_dataset_with_labels(torch.utils.data.Dataset):

    def __init__(self, timeseries, labels, past_nsample=48, future_nsample=24):

        self.past_nsample = past_nsample
        self.future_nsample = future_nsample

        self.data = torch.as_tensor(timeseries)
        self.labels = torch.as_tensor(labels)
        self.index = []

        ntraj = self.data.shape[0]
        nsample = self.data.shape[1]

        for traj_id in range(ntraj):
            for center in range(past_nsample, nsample - future_nsample):
                self.index.append(
                    (traj_id, center, self.labels[traj_id, center]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        traj_id, center, label = self.index[idx]

        x = self.data[traj_id]

        past = x[center - self.past_nsample:center]
        future = x[center:center + self.future_nsample]

        return (
            past, future,
            traj_id, center, label)


class encoder_dataset_no_labels(torch.utils.data.Dataset):

    def __init__(self, timeseries, past_nsample=48, future_nsample=24):

        self.past_nsample = past_nsample
        self.future_nsample = future_nsample

        self.data = torch.as_tensor(timeseries)
        self.index = []

        ntraj = self.data.shape[0]
        nsample = self.data.shape[1]

        for traj_id in range(ntraj):
            for center in range(past_nsample, nsample - future_nsample):
                self.index.append((traj_id, center))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        traj_id, center = self.index[idx]
        x = self.data[traj_id]

        past = x[center-self.past_nsample:center]
        future = x[center:center+self.future_nsample]

        return (
            past, future,
            traj_id, center)


class encoder(torch.nn.Module):
    """ Encodes future-consistent latent representation for subsequent regime classification. """

    def __init__(self, window_nsample, window_ndim, latent_ndim):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(window_nsample * window_ndim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_ndim)
        )

    def forward(self, x):
        return self.net(x)


class classifier_dataset_with_labels(torch.utils.data.Dataset):

    def __init__(self, signatures, labels, window_nsample=16):

        self.data = torch.as_tensor(signatures, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.window_nsample = window_nsample

        self.index = []
        ntraj = self.data.shape[0]
        nsample = self.data.shape[1]

        for traj_id in range(ntraj):
            for center in range(window_nsample, nsample - window_nsample):
                self.index.append(
                    (traj_id, center, self.labels[traj_id, center]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        traj_id, center, label = self.index[idx]
        window = self.data[traj_id, center-self.window_nsample:center+self.window_nsample]

        return (window, label)


class classifier_dataset_with_labels_and_negation(torch.utils.data.Dataset):

    def __init__(self, signatures, labels, window_nsample=16):

        self.data = torch.as_tensor(signatures, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.window_nsample = window_nsample

        self.index = []
        ntraj = self.data.shape[0]
        nsample = self.data.shape[1]

        for traj_id in range(ntraj):
            for center in range(window_nsample, nsample - window_nsample):

                negated = False
                self.index.append(
                    (traj_id, center, self.labels[traj_id, center], negated))

                negated = True
                self.index.append(
                    (traj_id, center, self.labels[traj_id, center], negated))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        traj_id, center, label, negated = self.index[idx]
        sign_factor = -1.0 if negated else 1.0
        window = sign_factor * self.data[traj_id, center-self.window_nsample:center+self.window_nsample]

        return (window, label)


class classifier_dataset_no_labels(torch.utils.data.Dataset):

    def __init__(self, signatures, window_nsample=16):

        self.data = torch.as_tensor(signatures, dtype=torch.float32)
        self.window_nsample = window_nsample

        self.index = []
        ntraj = self.data.shape[0]
        nsample = self.data.shape[1]

        for traj_id in range(ntraj):
            for center in range(window_nsample, nsample - window_nsample):
                self.index.append((traj_id, center))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        traj_id, center = self.index[idx]
        window = self.data[traj_id, center-self.window_nsample:center+self.window_nsample]

        return window
        

class classifier(torch.nn.Module):
    """ Classifies latent dynamical signatures to identify nominal and excursion regimes. """

    def __init__(self, input_ndim=3):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=input_ndim,
                out_channels=16,
                kernel_size=5,
                padding=2
            ),
            torch.nn.ReLU(),

            torch.nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Linear(32, 2)

    def forward(self, x):

        x = self.features(x)
        x = torch.mean(x, dim=-1)
        return self.classifier(x)


def compute_loss_slow(z, traj_id, center, temp_lag=1):
    loss = 0.0
    count = 0

    for traj in torch.unique(traj_id):

        idx = (traj_id == traj)
        z_traj = z[idx]
        c_traj = center[idx]

        order = torch.argsort(c_traj)
        z_traj = z_traj[order]
        dz = z_traj[temp_lag:] - z_traj[:-temp_lag]

        loss += (dz**2).mean()
        count += 1

    return loss / max(count,1)


def compute_loss_variance(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(torch.relu(1.0 - std))


def train_encoder(model_past, model_future, dataloader, temp_lag=1, nepoch=100, lr=1e-3):
    """Trains past and future encoder models in parallel."""

    optimizer = torch.optim.Adam(list(model_past.parameters()) + list(model_future.parameters()), lr=lr)
    mse_loss = torch.nn.MSELoss()

    print(f'[note]: training encoder models for {nepoch} epochs with {lr} learning rate')

    for epoch in range(nepoch):
        total_loss = 0.0

        # --! encoder training is unsupervised, i.e. there is no need for labels.
        # --! yet certain systems have labels, and thus the data loader can
        # --! return labels in the fifth element of the tuple. to account
        # --! for the rest of the systems that do not have labels, it
        # --! is important to explicitly ignore any element that
        # --! comes after the first four.
        for past, future, traj_id, center, *_ in dataloader:

            z = model_past(torch.flatten(past, start_dim=1))
            h = model_future(torch.flatten(future, start_dim=1))

            loss_future = mse_loss(z, h)
            loss_var_z = compute_loss_variance(z)
            loss_var_h = compute_loss_variance(h)

            loss_slow = compute_loss_slow(z[:, -1], traj_id, center, temp_lag=temp_lag)

            loss = (
                loss_future
                + 0.01 * loss_var_z + 0.01 * loss_var_h
                + 0.1 * loss_slow
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"\tepoch {epoch}, loss: {total_loss / len(dataloader):.6f}")

    print("")


def train_classifier(model, dataloader, nepoch=100, lr=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print(f'[note]: training classifier model for {nepoch} epochs with {lr} learning rate')

    for epoch in range(nepoch):
        total_loss = 0.0

        for x, y in dataloader:
            x = torch.transpose(x, -1, -2)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"\tepoch {epoch}, loss: {total_loss / len(dataloader):.6f}")

    print("")


class dataset_builder(interface):
    """Builds a dataset following the Builder design pattern."""

    @abstractmethod
    def add_labels(self, labels):
        pass

    @abstractmethod
    def set_negation(self, negated):
        pass

    @abstractmethod
    def build(self, data, past_nsample=48, future_nsample=24):
        pass


class encoder_dataset_builder(dataset_builder):
    """Builds an encoder dataset."""

    def __init__(self):
        self.labels = None

    def add_labels(self, labels):
        self.labels = labels
        return self

    def set_negation(self, negated):
        raise NotImplementedError()

    def build(self, data, past_nsample=48, future_nsample=24):
        if self.labels is not None:
            return encoder_dataset_with_labels(data, self.labels, past_nsample, future_nsample)
        else:
            return encoder_dataset_no_labels(data, past_nsample, future_nsample)


class classifier_dataset_builder(dataset_builder):
    """Builds a classifier dataset."""

    def __init__(self):
        self.labels = None
        self.negated = False

    def add_labels(self, labels):
        self.labels = labels
        return self

    def set_negation(self, negated):
        self.negated = negated
        return self

    def build(self, data, past_nsample=48, future_nsample=24):
        if self.labels is not None:
            if self.negated:
                return classifier_dataset_with_labels_and_negation(data, self.labels, past_nsample)
            else:
                return classifier_dataset_with_labels(data, self.labels, past_nsample)
        else:
            return classifier_dataset_no_labels(data, past_nsample)


def make_signatures(z, y=None, window_nsample=16):
    """
    Makes latent dynamical signatures out of future-consistent latent representation ``z`` and aligns result with labels ``y``.
    The alignment is defined by half-window length ``window_nsample``.
    """

    nsample = len(z)
    sigs = []
    labels = []

    for center in range(window_nsample, nsample - window_nsample):
        window = z[center-window_nsample:center+window_nsample]
        cov = np.cov(window.T)
        eigvals = np.linalg.eigvalsh(cov)

        sigs.append(eigvals)
        if y is not None: labels.append(y[center])

    return np.stack(sigs), np.stack(labels) if y is not None else None

