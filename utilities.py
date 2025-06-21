import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler as minmax_scaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns


def freeze_module(module):
    """Freezes all parameters in the given ``module``."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreezes all parameters in the given ``module``."""
    for param in module.parameters():
        param.requires_grad = True


def read_datafile(name: str, datachunk_len) -> torch.Tensor:
    """
    Reads data from a file called ``name`` and formats the data based on ``datachunk_len``,
    i.e. the length of one contiguous chunk of data. The file data are expected to be
    in format [T, C], such that the read data could be formatted as [B, T, C],
    where B, T and C are the number of batches, time steps and data channels,
    repectively.
    """

    # --! note that we force numpy loadtxt to return at least a two-dimensional array
    # --! by setting ndmin=2
    data = torch.tensor(
        np.loadtxt(name + '.csv', delimiter=',', dtype=np.float32, ndmin=2))
    datachunks_n = int(data.shape[0] / datachunk_len)

    # return read data in channels-last format
    return torch.reshape(data, (datachunks_n, datachunk_len, data.shape[1]))


def write_datafile(name: str, data, delim: str = ',') -> None:
    """Writes ``data`` to a file named ``name``. The file is written using a comma-separated-value format."""
    filedata = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    np.savetxt(name + '.csv', filedata, fmt='%.14f', delimiter=delim)


class fcnn(torch.nn.Module):
    """
    A fully-connected neural network.
    """
    def __init__(self, feat: list[int]=[1, 16 , 1], actfun_hid: str='relu', actfun_o: str='linear') -> None:
        """
        Constructs a fully-connected neural network with specified features ``feat`` and
        activation functions ``actfun_hid`` and ``actfun_o``.

        The features define the number of neurons in the network layers, e.g. a list
        of integers [1, 16, 1] describes a network that accepts a one-dimensional
        input, the network has one hidden layer with 16 neurons, and the
        network produces a one-dimensional output.

        The ``actfun_hid`` is a string name for activation functions in hidden layers,
        e.g. a string 'relu' translates into the torch class torch.nn.ReLU.
        The network will have the same ``actfun_hid`` in all hidden
        layers. For an output layer the activation function is
        specified using ``actfun_o``.
        
        Currently supported activation strings/functions are:
        'relu'    torch.nn.ReLU
        'tanh'    torch.nn.Tanh
        'sigmoid' torch.nn.Sigmoid
        'linear'  torch.nn.Identity
        """
        super().__init__()

        # use a helper constant to define the number of hidden layers
        nhid = len(feat) - 2

        # assemble a list of activation functions;
        # note that the number of hidden layers is incremented to accomodate the output layer,
        # so as long as we are counting hidden layers, these are set to user-specified
        # activation, but when we reach the output layer, it is set to user-defined string
        actfun = [actfun_hid if i < nhid else actfun_o for i in range(nhid + 1)]

        # construct a neural network;
        # note that the bias of all layers, the output included, is set to true
        self.net = torch.nn.Sequential(*[
            torch.nn.Sequential(*[
                torch.nn.Linear(i, o, bias=True), self._get_actfun(a)]) for i, o, a in zip(
                    feat[:-1], feat[1:], actfun)])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluates this neural network on ``data``."""
        return self.net(data)

    @staticmethod
    def _get_actfun(name: str):
        """
        Maps a string ``name`` that specifyes an activation function to a corresponding torch class object.
        """
        a = torch.nn.Identity
        if name == 'relu':
            a = torch.nn.ReLU
        elif name == 'tanh':
            a = torch.nn.Tanh
        elif name == 'sigmoid':
            a = torch.nn.Sigmoid
        elif name == 'linear':
            a = torch.nn.Identity
        else:
            raise ValueError(f'unknown activation function passed: {name}')
        return a()


def sample_timeseries(rng, sample_sz, timeseries_sz, timeseries_i, *timeseries_array):
    """
    Randomly samples timeseries of size ``sample_sz`` from given ``timeseries_array``. The array
    ``timeseries_array`` is indexed using ``timeseries_i``. Timeseries in ``timeseries_array``
    are expected to be shaped as one-dimensional arrays [T],
    where T is the number of timesteps.
    """

    # --! select current timeseries
    timeseries = timeseries_array[timeseries_i]

    # --! randomly locate a suitable sample region inside selected timeseries
    sample_start = int((timeseries_sz - sample_sz) * rng.random())
    sample_end   = sample_start + sample_sz
    sample       = timeseries[sample_start:sample_end, np.newaxis]

    # --! return a sample with mean removed
    return sample - np.mean(sample, axis=0, keepdims=True)


def next_timeseries_i(file_timeseries_i, timeseries_n):
    """Gets the next index of timeseries that are saved to a file."""
    return np.remainder(file_timeseries_i, timeseries_n)


def save_timeseries_train(timeseries, dir_name, snippet_sz):
    """
    Saves timeseries to files for training. Input ``timeseries`` is a list of column vectors, ``dir_name``
    specifies a directory name where files are saved, and ``snippet_sz`` is the length
    of saved timeseries snippets.
    """
    dataset_dir = dir_name

    data_config = [
        # number of timeseries in a file, file name
        (3500, 'train1'),
        (3500, 'train2'),
        (3500, 'train3'),
        (3500, 'train4'),
        (3500, 'train5'),
        (3500, 'train6'),
        (3500, 'train7'),
        (1000, 'valid'),
        (500,  'test')
    ]

    timeseries_n  = len(timeseries)
    timeseries_sz = len(timeseries[0][:, 0])

    if snippet_sz < timeseries_sz:
        for this, cfg in enumerate(data_config):

            # --! initialize a random number generator with a new seed
            rng = np.random.default_rng(seed=this + 1)

            data = np.stack([
                sample_timeseries(
                    rng,
                    snippet_sz,
                    timeseries_sz,
                    next_timeseries_i(file_timeseries_i, timeseries_n),
                    *timeseries) for file_timeseries_i in range(cfg[0])], axis=0)

            write_datafile(f'{dataset_dir}/{cfg[1]}', data)
        print('inf >> training files saved')
    else:
        print('err >> saved data size must be less than input timeseries size!')


def remove_mean(timeseries, dim: int=0):
    """Removes mean from ``timeseries`` in dimension ``dim``. Dimensions are preserved."""
    return timeseries - np.mean(timeseries, axis=dim, keepdims=True)


def save_timeseries_eval(timeseries, dir_name, snippet_sz):
    """
    Saves ``timeseries`` into folder named ``dir_name`` for model evaluation.
    The given ``timeseries`` are split into snippets, sized
    according to ``snippet_sz``.
    """
    timeseries_sz = len(timeseries[0][:, 0])

    snippets_n = timeseries_sz // snippet_sz
    if snippets_n > 0:
        timeseries_sz = snippet_sz * snippets_n

        # --! extract the actual saved timeseries and split them into snippets
        snippets = np.split(timeseries[0][:timeseries_sz, :1], snippets_n, axis=0)

        data = np.expand_dims(np.concatenate([remove_mean(snippet) for snippet in snippets], axis=0), 0)

        dataset_dir = dir_name
        filename    = 'eval'
        write_datafile(f'{dataset_dir}/{filename}', data)
        print('inf >> evaluation file saved')
    else:
        print('err >> saved timeseries sizes do not match!')


def scale_timeseries(timeseries):
    """
    Scales ``timeseries`` using min-max algorithm. The min-max range is -1 to 1, as this
    range should suit neural network training.
    """
    scaler = minmax_scaler(feature_range=(-1, 1))

    # --! format scaler input as column vectors
    scaler_inp = np.vstack([timeseries]).T
    return scaler.fit_transform(scaler_inp)


def train(model, parameters):
    """Trains a ``model`` with given ``parameters``."""

    dataset_dir           = parameters['dataset_dir']
    train_nfile           = parameters['train_nfile']
    timeseries_nsample    = parameters['timeseries_nsample']
    x_len                 = parameters['subtimeseries_nsample']
    alpha_fun             = parameters['alpha_fun']
    batsize               = parameters['batsize']
    nepoch                = parameters['nepoch']
    isverbose             = parameters['isverbose']
    isdmdonly             = parameters['isdmdonly']
    lr                    = parameters['learn_rate']
    weight_decay          = parameters['weight_decay']

    if isdmdonly:
        # --! we train the global operator here, so freeze the local one
        model.operator_sta.unfreeze()
        model.operator_dyn.freeze()

        # --! toggle the beta parameter defined in the paper
        model.fitweight_linearity_dmd = 1.
        model.fitweight_linearity_transformer  = 0.
    else:
        # --! we train the local operator now, so freeze the global one
        model.operator_sta.freeze()
        model.operator_dyn.unfreeze()

        # --! toggle the beta parameter defined in the paper
        model.fitweight_linearity_dmd = 0.
        model.fitweight_linearity_transformer  = 1.

    # --! specify optimizer
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay)

    # --! empty arrays to gather metrics
    loss_train_pred  = []
    loss_train_lin_g = []
    loss_train_lin_l = []
    loss_valid_pred  = []
    loss_valid_sta_mu = []
    valid_sta_sigma = []
    loss_valid_lin_g = []
    loss_valid_lin_l = []

    # --! prepare validation dataset
    data_valid    = read_datafile(f'{dataset_dir}/valid', timeseries_nsample)
    dataset_valid = torch.utils.data.TensorDataset(data_valid)

    # --! training duration
    if isverbose:
        print(f"inf >> Number of data files for training : {train_nfile}")

    for train_i in range(train_nfile):
        if isverbose:
            print(f"inf >> processing training file number {train_i + 1}")

        # --! make training datasets and loaders
        data_train = read_datafile(f'{dataset_dir}/train{train_i + 1}', timeseries_nsample)
        dataset_train = torch.utils.data.TensorDataset(data_train)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batsize, shuffle=True)

        # --! train
        for epoch in range(nepoch):

            # --! train neural networks
            for this, data in enumerate(dataloader_train):
                x = data[0][:, :x_len, :1]
                alpha = torch.zeros(x.shape[0], 1, 1) if alpha_fun is not None else torch.ones(x.shape[0], 1, 1)
                alpha = torch.round(alpha)

                optimizer.zero_grad()

                # --! fit a model to training data
                loss, loss_pred, loss_lin_g, loss_lin_l = model.fit(x, alpha)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    loss_train_pred.append(loss_pred)
                    loss_train_lin_g.append(loss_lin_g)
                    loss_train_lin_l.append(loss_lin_l)

            # --! validate results
            with torch.no_grad():
                dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batsize, shuffle=False)
                for data in dataloader_valid:
                    x = data[0][:, :x_len, :1]
                    alpha = torch.zeros(x.shape[0], 1, 1) if alpha_fun is not None else torch.ones(x.shape[0], 1, 1)

                    # --! validate prediction
                    model_o = model(x, alpha[:x.shape[0]])

                    timeseries_predict          = model_o[0]
                    sta_timeseries_predict_mu   = model_o[1]
                    sta_timseries_predict_sigma = model_o[2]
                    sta_fun                     = model_o[3]
                    sta_fun_predict             = model_o[4]
                    dyn_fun                     = model_o[5]
                    dyn_fun_predict             = model_o[6]
                    loss_valid_pred.append(torch.mean((x - timeseries_predict)**2))
                    loss_valid_sta_mu.append(torch.mean((x - sta_timeseries_predict_mu)**2))
                    valid_sta_sigma.append(torch.mean(sta_timseries_predict_sigma))
                    loss_valid_lin_g.append(torch.mean((sta_fun - sta_fun_predict)**2))
                    loss_valid_lin_l.append(torch.mean((dyn_fun - dyn_fun_predict)**2))

    o = (
        loss_train_pred,
        loss_train_lin_g, loss_train_lin_l,
        loss_valid_pred,
        loss_valid_lin_g, loss_valid_lin_l,
        loss_valid_sta_mu, valid_sta_sigma
    )

    return o


def test(model, parameters):
    """Tests a ``model`` on ``parameters``."""

    dataset_dir           = parameters['dataset_dir']
    timeseries_nsample    = parameters['timeseries_nsample']
    subtimeseries_nsample = parameters['subtimeseries_nsample']
    batsize               = parameters['batsize']
    alpha_fun             = parameters['alpha_fun']

    # --! make test datasets and loaders
    data_test = read_datafile(f'{dataset_dir}/test', timeseries_nsample)
    dataset_test = torch.utils.data.TensorDataset(data_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batsize, shuffle=False)

    loss_test_predict = []

    for data in dataloader_test:
        x = data[0][:, :subtimeseries_nsample, :1]
        alpha = torch.unsqueeze(alpha_fun(x), -1) if alpha_fun is not None else torch.ones(x.shape[0], 1, 1)

        o                   = model(x, alpha)
        timeseries_predict  = o[0]
        loss_test_predict.append(torch.mean((x - timeseries_predict)**2))

    return loss_test_predict


def disp_dataset(datadir, timeseries_nsample, timestep, ndata=3):
    """
    Displays metrics of a dataset located in a folder named ``datadir``. The size of timeseries
    stored in this dataset is defined by ``timeseries_nsample``. The ``timestep`` that
    was used when sampling the timeseries helps create time vectors for plotting.
    Finally, the number of timeseries to display is specified by ``ndata``.
    """

    # --! read data from files
    data_train = read_datafile(f'{datadir}/train1', timeseries_nsample)
    data_valid = read_datafile(f'{datadir}/valid', timeseries_nsample)
    data_test = read_datafile(f'{datadir}/test', timeseries_nsample)

    # --! compile dataset parameters
    data_table = [
        ( 'dataset',           'batches',        'timeseries length',          'channels'),
        ('--------',           '-------',        '-----------------',          '--------'),
        (   'train', data_train.shape[0], data_train.shape[1], data_train.shape[2]),
        (   'valid', data_valid.shape[0], data_valid.shape[1], data_valid.shape[2]),
        (    'test',  data_test.shape[0],  data_test.shape[1],  data_test.shape[2]) ]

    # --! print dataset parameters
    print('')
    print('inf >> dataset parameters:')
    print('')
    for row in data_table:
        print(f'{row[0]:>8} {row[1]:>8} {row[2]:>18} {row[3]:>8}')
    print('')

    t = torch.linspace(0., timestep*timeseries_nsample, timeseries_nsample)
    zero = torch.zeros_like(t)

    for i in range(ndata):
        data = data_train[i]

        plt.figure(figsize=(3, 3))
        plt.title(f'Data no. {i} from training dataset')
        plt.plot(t, data[:, 0], color='tab:blue', alpha=0.75, label='detuning')
        plt.plot(t, zero, color='tab:gray', linestyle='dotted', alpha=0.75)
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()


def disp_spectrum(eigvals):
    """Displays eigenvalues on the unit circle."""
    reals = eigvals.real.view(-1, 1)
    imags = eigvals.imag.view(-1, 1)

    plt.figure(figsize=(3, 3))
    plt.scatter(reals[:, 0], imags[:, 0], c='blue')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.title("Global Koopman operator spectrum")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary part")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def disp_spectrum_amps(model, dataset_dir, timeseries_nsample, timeseries_pos):
    """
    Displays the amplitudes of a ``model`` eigenvalues for data from ``dataset_dir`` indexed by
    ``timeseries_pos``. Parameter ``timeseries_nsample`` is needed to read ``dataset_dir``
    and extract proper timeseries. The displayed amplitudes are aligned
    with the corresponding ``model`` predictions.
    """

    eigval, eigvec        = torch.linalg.eig(model.operator_sta.model.weight)
    data_test             = read_datafile(f'{dataset_dir}/test', timeseries_nsample)
    subtimeseries_nsample = model.timeseries_nsample

    data_ic     = torch.unsqueeze(data_test[timeseries_pos][:model.param_kernsize, :1], 0)
    fun_ic      = model.operator_sta.embed(data_ic)
    eigvec_inv  = torch.linalg.inv(eigvec)
    fun_ic      = torch.squeeze(fun_ic, 0)
    eigvec_inv  = torch.squeeze(eigvec_inv, 0)
    fun_ic      = fun_ic.to(torch.cfloat)
    b           = torch.matmul(eigvec_inv, torch.transpose(fun_ic, 0, 1))
    b           = b.abs()
    b_nums      = np.array([range(len(b[:, 0]))]).reshape(-1, 1) + 1.0

    data        = data_test[timeseries_pos]
    timeseries  = torch.unsqueeze(data[:subtimeseries_nsample, :1], dim=0)
    o           = model(timeseries, alpha=1.0)

    sta_timeseries_predict_mu    = o[1]
    sta_timeseries_predict_sigma = o[2]
    timeseries                   = torch.squeeze(timeseries, dim=0)
    sta_timeseries_predict_mu    = torch.squeeze(sta_timeseries_predict_mu, dim=0)
    sta_timeseries_predict_sigma = torch.squeeze(sta_timeseries_predict_sigma, dim=0)

    sta_timeseries_predict_sigma = torch.exp(sta_timeseries_predict_sigma) + 1e-6

    var_max = torch.max(sta_timeseries_predict_sigma)
    var_max = 0.1 if var_max < 0.1 else var_max

    timestep = model.timestep
    t = np.arange(0., subtimeseries_nsample*timestep, timestep).reshape(-1, 1)

    plt.figure(figsize=(9,3))

    plt.subplot(1, 3, 1)
    plt.title('Mode amplitudes')
    plt.bar(b_nums[:, 0], b[:, 0])
    plt.xlabel('Mode index')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.title('Model response')
    plt.plot(t[:, 0], timeseries[:, 0], alpha=0.8, color='tab:green', label='$x$')
    plt.plot(t[:, 0], sta_timeseries_predict_mu[:, 0], alpha=1, color='tab:blue', linestyle='dashed', label='$\\mu(\\hat{x})$')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.title('Uncertainty')
    plt.plot(t[:, 0], sta_timeseries_predict_sigma[:, 0], alpha=1, color='tab:blue', label='$\\sigma^2$')
    plt.xlabel('Time [s]')
    plt.ylim((0., var_max))
    plt.legend()
    plt.tight_layout()

    plt.show()


def eval_model(model, alpha, datadir, timeseries_nsample, datasaved=False):
    """
    Evaluates a ``model`` on data from a folder named ``datadir``. Data read from
    that folder is split into timeseries according to ``timeseries_nsample``.
    When calling the ``model``, it is parameterized by ``alpha``.
    """
    data = read_datafile(f'{datadir}/eval', timeseries_nsample)

    # --! helping variables
    subtimeseries_nsample = model.timeseries_nsample
    timestep              = model.timestep
    timeseries_dur        = subtimeseries_nsample * timestep
    indeces               = range(data.shape[0])

    # --! data is a batch/array with timeseries, so split it along the batch dimension
    timeseries = torch.split(data, 1, dim=0)

    for k, x, a in zip(indeces, timeseries, alpha):

        # --! call the model
        o = model(x, a)

        mean    = o[0]
        logvar  = o[2]

        # --! remove the batch dimension
        x       = torch.squeeze(x, dim=0)
        mean    = torch.squeeze(mean, dim=0)
        logvar  = torch.squeeze(logvar, dim=0)

        var = torch.exp(logvar) + 1e-6

        # --! create a time vector
        t = np.arange(0., timeseries_dur, timestep).reshape(-1, 1)
        t = t + k*timeseries_dur

        # --! plot prediction result
        plt.figure(figsize=(6, 3))

        plt.subplot(1, 2, 1)
        plt.plot(t[:subtimeseries_nsample, 0], x[:subtimeseries_nsample, 0], alpha=0.8, color='tab:green', label='$x$')
        plt.plot(t[:subtimeseries_nsample, 0], mean[:, 0], alpha=1, color='tab:blue', linestyle='dashed', label='$\\mu(\\hat{x})$')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()

        var_max = torch.max(var)
        var_max = 0.1 if var_max < 0.1 else var_max

        plt.subplot(1, 2, 2)
        plt.plot(t[:subtimeseries_nsample, 0], var[:, 0], alpha=1, color='tab:blue', linestyle='solid', label='$\\sigma^2$')
        plt.xlabel('Time [s]')
        plt.ylim((0., var_max))
        plt.legend()
        plt.tight_layout()

        plt.show()

        if datasaved:
            savedata = np.expand_dims(np.concatenate([t, x, mean, var], axis=1), 0)
            write_datafile(f'savedata/statest_sim{k}', savedata, delim=' ')


def extract_poly_deg(polynomial: str='poly_1'):
    """Extracts a degree from the given ``polynomial`` string."""
    tokens = polynomial.split('_', 1)
    if len(tokens) != 2:
        raise ValueError('bad specification of a polynomial')
    deg = tokens[1]
    if not deg.isdigit():
        raise ValueError('bad specification of a polynomial')
    deg = int(tokens[1])
    if deg == 0:
        raise ValueError('zero degree polynomial is not supported')
    return deg


def label_stationarity(dmd_model, dmd_residual_max, dataset_dir, data_timeseries_sz):
    """
    Based on DMD model residuals, creates a set of labels for stationary/non-stationary data.
    """
    data = read_datafile(f'{dataset_dir}/test', data_timeseries_sz)
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

