import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler as minmax_scaler
from matplotlib import pyplot as plt


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
        np.loadtxt('./data/' + name + '.csv', delimiter=',', dtype=np.float32, ndmin=2))
    datachunks_n = int(data.shape[0] / datachunk_len)

    # return read data in channels-last format
    return torch.reshape(data, (datachunks_n, datachunk_len, data.shape[1]))


def write_datafile(name: str, data, delim: str = ',') -> None:
    """Writes ``data`` to a file named ``name``. The file is written using a comma-separated-value format."""
    filedata = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    np.savetxt('./data/' + name + '.csv', filedata, fmt='%.14f', delimiter=delim)


class fcnn(torch.nn.Module):
    """
    A fully-connected neural network.
    """
    def __init__(self, features: list[int]=[1, 16 , 1], act_fn_hidden: str='relu', act_fn_out: str='linear') -> None:
        """
        Constructs a fully-connected neural network with specified ``features`` and ``activation``.

        The ``features`` define the number of neurons in the network layers, e.g. a list
        of integers [1, 16, 1] describes a network that accepts a one-dimensional
        input, the network has one hidden layer with 16 neurons, and the
        network produces a one-dimensional output.

        The ``act_fn_hidden`` is a string name for activation functions, e.g. a string 'relu'
        translates into the torch class torch.nn.ReLU. The network will have one
        single ``act_fn_hidden`` everywhere, except for an output layer -
        this one can be specified using ``act_fn_out``. Currently
        supported activation strings/functions are:
        'relu'    torch.nn.ReLU
        'tanh'    torch.nn.Tanh
        'linear'  torch.nn.Identity
        """
        super().__init__()

        # use a helper constant to define the number of hidden layers
        hidden_n = len(features) - 2

        # assemble a list of activation functions;
        # note that the number of hidden layers is incremented to accomodate the output layer,
        # so as long as we are counting hidden layers, these are set to user-specified
        # activation, but when we reach the output layer, it is set to user-defined string
        activations = [act_fn_hidden if i < hidden_n else act_fn_out for i in range(hidden_n + 1)]

        # construct a neural network;
        # note that the bias of all layers, the output included, is set to true
        self.net = torch.nn.Sequential(*[
            torch.nn.Sequential(*[
                torch.nn.Linear(i, o, bias=True), self.get_activation_by_name(a)]) for i, o, a in zip(
                    features[:-1], features[1:], activations)])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluates this neural network on ``data``."""
        return self.net(data)

    @staticmethod
    def get_activation_by_name(name: str):
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
    dataset_dir = 'cavity/' + dir_name

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

        dataset_dir = 'cavity/' + dir_name
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

    dataset_dir       = parameters['dataset_dir']
    datafiles_train_n = parameters['train_files_n']
    timeseries_len    = parameters['timeseries_sz']
    bat_sz            = parameters['batch_sz']
    epochs_n          = parameters['epochs_n']
    x_len             = parameters['x_sz']
    verbose           = parameters['is_verbose']
    global_on         = parameters['is_global']
    lr                = parameters['learn_rate']
    weight_decay      = parameters['weight_decay']
    alpha             = parameters['alpha']

    if global_on:
        # --! we train the global operator here, so freeze the local one
        unfreeze_module(model.fun_params_kern_enc_g)
        freeze_module(model.fun_params_kern_enc_l)

        unfreeze_module(model.timeseries_dyn)
        freeze_module(model.funs_dyn_enc)
        freeze_module(model.funs_dyn)

        unfreeze_module(model.dec_g)
        freeze_module(model.dec_l)

        # --! toggle the beta parameter defined in the paper
        model.fit_weight_lin_global = 1.
        model.fit_weight_lin_local  = 0.
    else:
        # --! we train the local operator now, so freeze the global one
        freeze_module(model.fun_params_kern_enc_g)
        unfreeze_module(model.fun_params_kern_enc_l)

        freeze_module(model.timeseries_dyn)
        unfreeze_module(model.funs_dyn_enc)
        unfreeze_module(model.funs_dyn)

        freeze_module(model.dec_g)
        unfreeze_module(model.dec_l)

        # --! toggle the beta parameter defined in the paper
        model.fit_weight_lin_global = 0.
        model.fit_weight_lin_local  = 1.

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
    loss_valid_lin_g = []
    loss_valid_lin_l = []

    # --! prepare validation dataset
    data_valid    = read_datafile(f'{dataset_dir}/valid', timeseries_len)
    dataset_valid = torch.utils.data.TensorDataset(data_valid)

    # --! training duration
    if verbose:
        print(f"inf >> Number of data files for training : {datafiles_train_n}")

    for datafile_train in range(datafiles_train_n):
        if verbose:
            print(f"inf >> processing training file number {datafile_train + 1}")

        # --! make training datasets and loaders
        data_train = read_datafile(f'{dataset_dir}/train{datafile_train + 1}', timeseries_len)
        dataset_train = torch.utils.data.TensorDataset(data_train)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=bat_sz, shuffle=True)

        # --! train
        for epoch in range(epochs_n):

            # --! train neural networks
            for this, data in enumerate(dataloader_train):
                x = data[0][:, :x_len, :1]

                optimizer.zero_grad()

                # --! fit a model to training data
                loss, loss_pred, loss_lin_g, loss_lin_l = model.fit(x, global_only=global_on, fixed_alpha=alpha)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    loss_train_pred.append(loss_pred)
                    loss_train_lin_g.append(loss_lin_g)
                    loss_train_lin_l.append(loss_lin_l)

            # --! validate results
            with torch.no_grad():
                dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=bat_sz, shuffle=False)
                for data in dataloader_valid:
                    x  = data[0][:, :x_len, :1] # take only displacement

                    # --! validate prediction
                    outs = model(x, alpha=alpha)
                    funs_g          = outs[0]
                    funs_g_pred     = outs[1]
                    funs_l          = outs[2]
                    funs_l_pred     = outs[3]
                    timeseries_pred = outs[4]
                    loss_valid_pred.append(torch.mean((x - timeseries_pred)**2))
                    loss_valid_lin_g.append(torch.mean((funs_g - funs_g_pred)**2))
                    loss_valid_lin_l.append(torch.mean((funs_l - funs_l_pred)**2))

    return loss_train_pred, loss_train_lin_g, loss_train_lin_l, loss_valid_pred, loss_valid_lin_g, loss_valid_lin_l


def test(model, parameters):
    """Tests a ``model`` on ``parameters``."""

    dataset_dir       = parameters['dataset_dir']
    timeseries_len    = parameters['timeseries_sz']
    bat_sz            = parameters['batch_sz']
    x_len             = parameters['x_sz']
    alpha             = parameters['alpha']

    # --! make test datasets and loaders
    data_test = read_datafile(f'{dataset_dir}/test', timeseries_len)
    dataset_test = torch.utils.data.TensorDataset(data_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bat_sz, shuffle=False)

    loss_test_pred = []

    for data in dataloader_test:
        x = data[0][:, :x_len, :1] # detuning is one-dimensional

        outs = model(x, alpha=alpha)
        funs_g          = outs[0]
        funs_g_pred     = outs[1]
        funs_l          = outs[2]
        funs_l_pred     = outs[3]
        timeseries_pred = outs[4]
        loss_test_pred.append(torch.mean((x - timeseries_pred)**2))

    return loss_test_pred


def disp_dataset(dataset_dir, timeseries_sz, timestep, data_n=3):
    """
    Displays metrics of a dataset located in a folder named ``dataset_dir``. The size of timeseries
    stored in this dataset is defined by ``timeseries_sz``. The ``timestep`` that was used
    when sampling the timeseries helps create time vectors for plotting.
    """

    # --! read data from files
    data_train = read_datafile(f'{dataset_dir}/train1', timeseries_sz)
    data_valid = read_datafile(f'{dataset_dir}/valid', timeseries_sz)
    data_test = read_datafile(f'{dataset_dir}/test', timeseries_sz)

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

    t = torch.linspace(0., timestep*timeseries_sz, timeseries_sz)
    zero = torch.zeros_like(t)

    for i in range(data_n):
        data = data_train[i]

        plt.figure(figsize=(4, 4))
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

    plt.figure(figsize=(4, 4))
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


def disp_spectrum_amps(model, dataset_dir, data_timeseries_sz, data_i):
    """
    Displays the amplitudes of a ``model`` eigenvalues for data from ``dataset_dir`` indexed
    by ``data_i``. Parameter ``data_timeseries_sz`` is needed to read ``dataset_dir``
    and extract proper timeseries. The displayed amplitudes are aligned
    with the corresponding ``model`` predictions.
    """

    eigvals, eigvecs = torch.linalg.eig(model.timeseries_dyn.weight)
    data_test        = read_datafile(f'{dataset_dir}/test', data_timeseries_sz)
    x_len            = model.timeseries_sz

    data_ic     = torch.unsqueeze(data_test[data_i][:model.fun_params_kern_sz, :1], 0)
    funs_ic     = model._embed_functions_g(data_ic)
    eigvecs_inv = torch.linalg.inv(eigvecs)
    funs_ic     = torch.squeeze(funs_ic, 0)
    eigvecs_inv = torch.squeeze(eigvecs_inv, 0)
    funs_ic     = funs_ic.to(torch.cfloat)
    b           = torch.matmul(eigvecs_inv, torch.transpose(funs_ic, 0, 1))
    b           = b.abs()
    b_nums      = np.array([range(len(b[:, 0]))]).reshape(-1, 1) + 1.0

    data        = data_test[data_i]
    timeseries  = torch.unsqueeze(data[:x_len, :1], dim=0)
    outs        = model(timeseries, alpha=1.0)
    timeseries_pred = outs[4]
    timeseries      = torch.squeeze(timeseries, dim=0)
    timeseries_pred = torch.squeeze(timeseries_pred, dim=0)

    timestep = model.timeseries_timestep
    t = np.arange(0., x_len*timestep, timestep).reshape(-1, 1)

    plt.figure(figsize=(7,4))
    plt.subplot(1, 2, 1)

    plt.bar(b_nums[:, 0], b[:, 0])
    plt.title('Mode amplitudes')
    plt.xlabel('Mode index')
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.plot(t[:, 0], timeseries[:, 0], alpha=0.8, color='tab:green', label='$x$')
    plt.plot(t[:, 0], timeseries_pred[:, 0], alpha=1, color='tab:blue', linestyle='dashed', label='$\\hat{x}$')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.tight_layout()
    plt.show()
