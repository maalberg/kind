import torch

import utilities as util

def train(model, param):

    # --!--------------------------------------------------------------------!
    # --! initialize common variables
    # --!--------------------------------------------------------------------!

    timeseries_nsample    = param['timeseries_nsample']
    subtimeseries_nsample = param['subtimeseries_nsample']
    train_nfile           = param['train_nfile']
    nepoch                = param['nepoch']
    batsize               = param['batsize']
    alphafun              = param['alphafun']
    learnrate             = param['learnrate']
    weightdecay           = param['weightdecay']
    isstaonly             = param['isstaonly']
    ismeanonly            = param['ismeanonly']
    isverbose             = param['isverbose']

    # --!--------------------------------------------------------------------!
    # --! configure the model
    # --!--------------------------------------------------------------------!

    #sta_varloss = None

    #if isstaonly:
        # --! we train the stationary operator, so freeze the adaptive one
        #model.operator_dyn.freeze_mean()

        #if ismeanonly:
            # --! for the first phase of training, freeze variance
            #model.operator_sta.unfreeze()
            #model.operator_sta.freeze_var()
            #sta_varloss = False

        #else:
            # --! for the second phase, freeze mean
            #model.operator_sta.unfreeze()
            #model.operator_sta.freeze_mean()
            #sta_varloss = True

    #else:
        #raise NotImplementedError

    # --!--------------------------------------------------------------------!
    # --! select current datasets
    # --!--------------------------------------------------------------------!

    datadir = None

    if isstaonly:
        if ismeanonly:
            datadir = param['stadatadir']
 
        else:
            datadir = param['mixdatadir']

    else:
        datadir = param['transdatadir']

    # --! prepare test data
    testdata    = util.read_datafile(f'{datadir}/valid', timeseries_nsample)
    testdataset = torch.utils.data.TensorDataset(testdata)

    # --!--------------------------------------------------------------------!
    # --! run a training loop
    # --!--------------------------------------------------------------------!

    # --! training duration
    if isverbose: print(f"inf >> number of data files for training is {train_nfile}")

    trainloss_predict    = []
    trainloss_sta_lin    = []
    trainloss_dyn_lin    = []

    testloss_predict     = []
    testloss_sta_mean    = []
    testloss_sta_var     = []
    testloss_sta_lin     = []
    testloss_dyn_lin     = []
    testloss_dyn         = []

    if model.fit_next():

        # --! specify an optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learnrate,
            weight_decay=weightdecay)

        for ifile in range(train_nfile):
            if isverbose: print(f"inf >> processing training file number {ifile + 1}")

            # --! prepare training data
            traindata     = util.read_datafile(f'{datadir}/train{ifile + 1}', timeseries_nsample)
            traindataset  = torch.utils.data.TensorDataset(traindata)
            traindatafun  = torch.utils.data.DataLoader(traindataset, batch_size=batsize, shuffle=True)

            # --! train
            for epoch in range(nepoch):
                for data in traindatafun:
                    timeseries = data[0][:, :subtimeseries_nsample, :1]
                    #alpha      = torch.zeros(timeseries.shape[0], 1, 1) if alphafun is not None else torch.ones(timeseries.shape[0], 1, 1)

                    optimizer.zero_grad()

                    # --! fit a model to training time series
                    loss, loss_predict, loss_lin_g, loss_lin_l = model.fit(timeseries)

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        trainloss_predict.append(loss_predict)
                        trainloss_sta_lin.append(loss_lin_g)
                        trainloss_dyn_lin.append(loss_lin_l)

                # --! test
                with torch.no_grad():
                    testdatafun = torch.utils.data.DataLoader(testdataset, batch_size=batsize, shuffle=False)
                    for data in testdatafun:
                        timeseries = data[0][:, :subtimeseries_nsample, :1]
                        alpha      = torch.zeros(timeseries.shape[0], 1, 1) if alphafun is not None else torch.ones(timeseries.shape[0], 1, 1)

                        # --! test prediction
                        model_o = model(timeseries, alpha)

                        timeseries_predict          = model_o[0]
                        sta_timeseries_predict_mean = model_o[1]
                        sta_timeseries_predict_var  = model_o[2]
                        dyn_timeseries_predict      = model_o[3]
                        sta_fun                     = model_o[4]
                        sta_fun_predict             = model_o[5]
                        dyn_fun                     = model_o[6]
                        dyn_fun_predict             = model_o[7]

                        testloss_predict.append(torch.mean((timeseries - timeseries_predict)**2))
                        testloss_sta_mean.append(torch.mean((timeseries - sta_timeseries_predict_mean)**2))
                        testloss_sta_var.append(torch.mean(sta_timeseries_predict_var))
                        testloss_dyn.append(torch.mean((timeseries - dyn_timeseries_predict)**2))
                        testloss_sta_lin.append(torch.mean((sta_fun - sta_fun_predict)**2))
                        testloss_dyn_lin.append(torch.mean((dyn_fun - dyn_fun_predict)**2))

    o = (
        trainloss_predict,
        trainloss_sta_lin, trainloss_dyn_lin,
        testloss_predict,
        testloss_sta_lin, testloss_dyn_lin,
        testloss_sta_mean, testloss_sta_var,
        testloss_dyn
    )

    return o