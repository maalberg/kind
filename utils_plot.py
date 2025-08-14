# --!--------------------------------------------------------------!
# --! utilities for plotting
# --!--------------------------------------------------------------!

import torch
import numpy as np
from matplotlib import pyplot as plt

import utils_data


def plot_dataset(datadir, timeseries_nsample, timestep):
    """
    Displays metrics of a dataset located in a folder named ``datadir``. The size of timeseries
    stored in this dataset is defined by ``timeseries_nsample``. The ``timestep`` that
    was used when sampling the timeseries helps create time vectors for plotting.
    """

    # --! read data from files
    data_train = utils_data.read_datafile(f'{datadir}/train1', timeseries_nsample)
    data_valid = utils_data.read_datafile(f'{datadir}/valid', timeseries_nsample)
    data_test  = utils_data.read_datafile(f'{datadir}/test', timeseries_nsample)

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

    # --! make time array and a helping line to demarcate a zero level on a plot
    t = torch.linspace(0., timestep*timeseries_nsample, timeseries_nsample)
    zero = torch.zeros_like(t)

    # --! show two examples for each channel
    ndata = 2

    # --! limit the number of displayed channels
    nchannel = 3 if data_train.shape[2] > 3 else data_train.shape[2]

    sub_w = 3
    sub_h = 3
    fig_w = nchannel * sub_w
    fig_h = ndata * sub_h

    plt.figure(figsize=(fig_w, fig_h))

    for j in range(ndata):
        data = data_train[j]

        for k in range(nchannel):
            plt.subplot(ndata, nchannel, j*ndata + k + 1)
            if j==0: plt.title(f'Detuning index {k}')
            plt.plot(t, data[:, k], color='tab:blue', alpha=0.75)
            plt.plot(t, zero, color='tab:gray', linestyle='dotted', alpha=0.75)
            if j == ndata - 1: plt.xlabel('Time [s]')
            if k == 0: plt.ylabel('Amplitude')
            plt.tight_layout()

    plt.show()


def plot_eigs(model):
    """Displays eigenvalues of given ``model`` on the unit circle."""
    print(model.operator_stat.mod_mean.weight)

    eigvals, _ = torch.linalg.eig(model.operator_stat.mod_mean.weight)

    reals = eigvals.real.view(-1, 1)
    imags = eigvals.imag.view(-1, 1)

    plt.figure(figsize=(3, 3))
    plt.scatter(reals[:, 0], imags[:, 0], c='blue')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.title("Stationary operator spectrum")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary part")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_modes(model, datadir, timeseries_nsample, jtimeseries):
    """
    Displays the amplitudes of a ``model`` eigenvalues for data from ``datadir`` indexed by
    ``jtimeseries``. Parameter ``timeseries_nsample`` is needed to read ``datadir``
    and extract proper timeseries. The displayed amplitudes are aligned
    with the corresponding ``model`` predictions.
    """

    # --! extract eigenvalues and eigenvectors from a stationary DMD-like operator
    eigval, eigvec        = torch.linalg.eig(model.operator_stat.mod_mean.weight)
    testdata              = utils_data.read_datafile(f'{datadir}/test', timeseries_nsample)
    lookback_nsample      = model.lookback_nsample
    timeseries_ndim       = model.timeseries_ndim

    # --! take the initial condition of timeseries specified by index j and
    # --! embed this initial condition into the latent space
    # --! of the stationary model
    data_ic     = torch.unsqueeze(testdata[jtimeseries][:model.param_kernsize, :], 0)
    fun_ic      = model.operator_stat.embed(data_ic)

    # --! now multiply eigenvectors and initial conditions together in a dot product fashion to
    # --! find out how aligned these two are, and thus we get our modal amplitude,
    # --! where a greater modal amplitude means more 'involvement' of a
    # --! particular eigenvalue in modeling the evolution of
    # --! particular time series.
    #
    # --! under the hood, the eignevectors must be inverted to achieve proper projection
    # --! of the initial condition into the eigen basis
    eigvec_inv  = torch.linalg.inv(eigvec)
    fun_ic      = torch.squeeze(fun_ic, 0)
    eigvec_inv  = torch.squeeze(eigvec_inv, 0)
    fun_ic      = fun_ic.to(torch.cfloat)
    amp         = torch.matmul(eigvec_inv, torch.transpose(fun_ic, 0, 1))

    # --! modal amplitudes are calculated as complex numbers, but we want only the real part
    amp         = amp.abs()
    jamp        = np.array([range(len(amp[:, 0]))]).reshape(-1, 1) + 1.0

    data        = testdata[jtimeseries]
    timeseries  = torch.unsqueeze(data, dim=0)
    model_i     = timeseries[:, :lookback_nsample]
    model_o     = model.operator_stat(model_i)

    timeseries_pre_mean   = model_o[0]
    timeseries_pre_logvar = model_o[1]

    timeseries            = torch.squeeze(timeseries, dim=0)
    timeseries_pre_mean   = torch.squeeze(timeseries_pre_mean, dim=0)
    timeseries_pre_logvar = torch.squeeze(timeseries_pre_logvar, dim=0)

    timeseries_pre_var    = torch.exp(timeseries_pre_logvar) + 1e-6

    var_max = torch.max(timeseries_pre_var)
    var_max = 0.1 if var_max < 0.1 else var_max

    timestep = model.timestep
    t = np.arange(0., timeseries_nsample*timestep, timestep).reshape(-1, 1)

    plt.figure(figsize=(9,3))

    plt.subplot(1, 3, 1)
    plt.title('Mode amplitudes')
    plt.bar(jamp[:, 0], amp[:, 0])
    plt.xlabel('Mode index')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.title('Model response')
    for k in range(timeseries_ndim):
        plt.plot(t[:, 0], timeseries[:, k], alpha=0.8, label='$x_{' + f'{k+1}' + '}$')
        plt.plot(t[:, 0], timeseries_pre_mean[:, k], alpha=1, linestyle='dashed', label='$\\mu(\\hat{x_{' + f'{k+1}' + '}})$')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.title('Uncertainty')
    for k in range(timeseries_ndim):
        plt.plot(t[:, 0], timeseries_pre_var[:, k], alpha=1, label='$\\zeta_{' f'{k+1}' + '}$')
    plt.xlabel('Time [s]')
    plt.ylim((0., var_max))
    plt.legend()
    plt.tight_layout()

    plt.show()


def plot_stationary(model, datadir, timeseries_nsample, datasaved=False):
    """
    Plots the results of stationary ``model`` evaluation, including the mean and variance
    of stationary ``model`` predictions. The data for ``model`` evaluation is read
    from a directory, called ``datadir``. The read data is shaped into
    timeseries according to the number of samples, specified in
    ``timeseries_nsample``. Plotted results can also be saved
    to files if ``datasaved`` flag is set to True.
    """
    data = utils_data.read_datafile(f'{datadir}/eval', timeseries_nsample)

    # --! helping variables
    timestep              = model.timestep
    timeseries_dur        = timeseries_nsample * timestep
    timeseries_ndim       = model.timeseries_ndim
    lookback_nsample      = model.lookback_nsample
    indeces               = range(data.shape[0])

    # --! data is a batch/array with timeseries, so split it along the batch dimension
    timeseries = torch.split(data, 1, dim=0)

    for j, x in zip(indeces, timeseries):

        # --! extract the lookback window
        model_i = x[:, :lookback_nsample]

        # --! call the model
        o      = model(model_i)
        mean   = o[1]
        logvar = o[2]

        # --! remove the batch dimension
        x      = torch.squeeze(x, dim=0)
        mean   = torch.squeeze(mean, dim=0)
        logvar = torch.squeeze(logvar, dim=0)

        # --! convert log-variance to variance
        var    = torch.exp(logvar) + 1e-6

        # --! create a time vector
        t = np.arange(0., timeseries_dur, timestep).reshape(-1, 1)
        t = t + j*timeseries_dur

        # --! shift the forecast begin to the current window
        t_forecast_begin = timestep * lookback_nsample + j*timeseries_dur
        mean_min = torch.min(mean)
        mean_max = torch.max(mean)

        # --! plot prediction result
        plt.figure(figsize=(6, 3))

        plt.subplot(1, 2, 1)
        plt.plot([t_forecast_begin, t_forecast_begin], [mean_min, mean_max], linestyle='dotted', color='gray')
        for k in range(timeseries_ndim):
            plt.plot(t[:, 0], x[:, k], alpha=0.8, label='$x_{' + f'{k+1}' + '}$')
            plt.plot(t[:, 0], mean[:, k], alpha=1, linestyle='dashed', label='$\\mu(\\hat{x_{' + f'{k+1}' + '}})$')
        plt.ylabel('Amplitude')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.tight_layout()

        maxvar = torch.max(var)
        maxvar = 0.1 if maxvar < 0.1 else maxvar

        plt.subplot(1, 2, 2)
        plt.plot([t_forecast_begin, t_forecast_begin], [mean_min, mean_max], linestyle='dotted', color='gray')
        for k in range(timeseries_ndim):
            plt.plot(t[:, 0], var[:, k], alpha=1, linestyle='solid', label='$\\zeta_{' f'{k+1}' + '}$')
        plt.xlabel('Time [s]')
        plt.ylim((0., maxvar))
        plt.legend()
        plt.tight_layout()

        plt.show()

        if datasaved:
            savedata = np.expand_dims(np.concatenate([t, x, mean, var], axis=1), 0)
            utils_data.write_datafile(f'savedata/statest_sim{k}', savedata, delim=' ')


def plot_transient(model, datadir, timeseries_nsample, datasaved=False):
    """
    Plots the transient response of the given ``model`` to data read from a directory,
    called ``datadir``.
    """
    data = utils_data.read_datafile(f'{datadir}/eval', timeseries_nsample)

    # --! helping variables
    lookback_nsample      = model.lookback_nsample
    timestep              = model.timestep
    timeseries_ndim       = model.timeseries_ndim
    timeseries_dur        = timeseries_nsample * timestep
    indeces               = range(data.shape[0])

    # --! data is a batch/array with timeseries, so split it along the batch dimension
    timeseries = torch.split(data, 1, dim=0)

    for j, x in zip(indeces, timeseries):

        # --! extract the lookback window
        model_i = x[:, :lookback_nsample]

        # --! call the model
        o      = model(model_i)
        mean   = o[3]
        logvar = o[4]

        # --! remove the batch dimension
        x      = torch.squeeze(x, dim=0)
        mean   = torch.squeeze(mean, dim=0)
        logvar = torch.squeeze(logvar, dim=0)

        # --! convert log-variance to variance
        var    = torch.exp(logvar) + 1e-6

        # --! create a time vector
        t = np.arange(0., timeseries_dur, timestep).reshape(-1, 1)
        t = t + j*timeseries_dur

        # --! shift the forecast begin to the current window
        t_forecast_begin = timestep * lookback_nsample + j*timeseries_dur
        mean_min = torch.min(mean)
        mean_max = torch.max(mean)

        # --! plot prediction result
        plt.figure(figsize=(6, 3))

        plt.subplot(1, 2, 1)
        plt.plot([t_forecast_begin, t_forecast_begin], [mean_min, mean_max], linestyle='dotted', color='gray')
        for k in range(timeseries_ndim):
            plt.plot(t[:, 0], x[:, k], alpha=0.8, label='$x_{' + f'{k+1}' + '}$')
            plt.plot(t[:, 0], mean[:, k], alpha=1, linestyle='dashed', label='$\\mu(\\hat{x_{' + f'{k+1}' + '}})$')
        plt.ylabel('Amplitude')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.tight_layout()

        maxvar = torch.max(var)
        maxvar = 0.1 if maxvar < 0.1 else maxvar

        plt.subplot(1, 2, 2)
        plt.plot([t_forecast_begin, t_forecast_begin], [mean_min, mean_max], linestyle='dotted', color='gray')
        for k in range(timeseries_ndim):
            plt.plot(t[:, 0], var[:, k], alpha=1, linestyle='solid', label='$\\zeta_{' f'{k+1}' + '}$')
        plt.xlabel('Time [s]')
        plt.ylim((0., maxvar))
        plt.legend()
        plt.tight_layout()

        plt.show()

        if datasaved:
            savedata = np.expand_dims(np.concatenate([t, x, mean, var], axis=1), 0)
            utils_data.write_datafile(f'savedata/dyntest_sim{k}', savedata, delim=' ')


def plot_blend(model, datadir, timeseries_nsample, datasaved=False):
    data = utils_data.read_datafile(f'{datadir}/eval', timeseries_nsample)

    # --! helping variables
    lookback_nsample      = model.lookback_nsample
    timestep              = model.timestep
    timeseries_ndim       = model.timeseries_ndim
    timeseries_dur        = timeseries_nsample * timestep
    indeces               = range(data.shape[0])

    # --! data is a batch/array with timeseries, so split it along the batch dimension
    timeseries = torch.split(data, 1, dim=0)

    for j, x in zip(indeces, timeseries):

        # --! extract the lookback window
        model_i = x[:, :lookback_nsample]

        # --! call the model
        o             = model(model_i)
        mean          = o[0]
        stat_logvar   = o[2]
        trans_logvar  = o[4]
        alpha         = o[9]

        # --! remove the batch dimension
        x            = torch.squeeze(x, dim=0)
        mean         = torch.squeeze(mean, dim=0)
        stat_logvar  = torch.squeeze(stat_logvar, dim=0)
        trans_logvar = torch.squeeze(trans_logvar, dim=0)
        alpha        = torch.squeeze(alpha, dim=0)

        # --! convert log-variance to variance
        stat_var     = torch.exp(stat_logvar) + 1e-6
        trans_var    = torch.exp(trans_logvar) + 1e-6

        # --! create a time vector
        t = np.arange(0., timeseries_dur, timestep).reshape(-1, 1)
        t = t + j*timeseries_dur

        # --! shift the forecast begin to the current window
        t_forecast_begin = timestep * lookback_nsample + j*timeseries_dur
        mean_min = torch.min(mean)
        mean_max = torch.max(mean)

        # --! plot prediction result
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 4, 1)
        plt.plot([t_forecast_begin, t_forecast_begin], [mean_min, mean_max], linestyle='dotted', color='gray')
        for k in range(timeseries_ndim):
            plt.plot(t[:, 0], x[:, k], alpha=0.8, label='$x_{' + f'{k+1}' + '}$')
            plt.plot(t[:, 0], mean[:, k], alpha=1, linestyle='dashed', label='$\\mu(\\hat{x_{' + f'{k+1}' + '}})$')
        plt.ylabel('Amplitude')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.tight_layout()

        plt.subplot(1, 4, 2)
        plt.plot([t_forecast_begin, t_forecast_begin], [mean_min, mean_max], linestyle='dotted', color='gray')
        for k in range(timeseries_ndim):
            plt.plot(t[:, 0], alpha[:, k], alpha=1, linestyle='solid', label='$\\alpha_{' + f'{k+1}' + '}$')
        plt.xlabel('Time [s]')
        plt.ylim((0., 1.))
        plt.xlabel('Time [s]')
        plt.legend()
        plt.tight_layout()

        maxvar = torch.max(stat_var)
        maxvar = 0.1 if maxvar < 0.1 else maxvar

        plt.subplot(1, 4, 3)
        plt.plot([t_forecast_begin, t_forecast_begin], [mean_min, mean_max], linestyle='dotted', color='gray')
        for k in range(timeseries_ndim):
            plt.plot(t[:, 0], stat_var[:, k], alpha=1, linestyle='solid', label='$\\zeta_{' f'{k+1}' + '}$')
        plt.xlabel('Time [s]')
        plt.ylim((0., maxvar))
        plt.legend()
        plt.tight_layout()

        maxvar = torch.max(trans_var)
        maxvar = 0.1 if maxvar < 0.1 else maxvar

        plt.subplot(1, 4, 4)
        plt.plot([t_forecast_begin, t_forecast_begin], [mean_min, mean_max], linestyle='dotted', color='gray')
        for k in range(timeseries_ndim):
            plt.plot(t[:, 0], trans_var[:, k], alpha=1, linestyle='solid', label='$\\zeta_{' f'{k+1}' + '}$')
        plt.xlabel('Time [s]')
        plt.ylim((0., maxvar))
        plt.legend()
        plt.tight_layout()

        plt.show()

        if datasaved:
            savedata = np.expand_dims(np.concatenate([t, x, mean, stat_var, trans_var, alpha], axis=1), 0)
            utils_data.write_datafile(f'savedata/blendtest_sim{k}', savedata, delim=' ')
