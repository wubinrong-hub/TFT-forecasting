# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:05:09 2024

@author: DELL
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import warnings
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
warnings.filterwarnings("ignore")


data = pd.read_csv("data.csv")
data["id"] = "ChinaDADOU"
data["time_idx"] = pd.to_datetime(data["data"]).astype(np.int64)
data["time_idx"] -= data["time_idx"].min()
data["time_idx"] = (data.time_idx / 3600000000000) + 1
data["time_idx"] = data["time_idx"].astype(int)
data1 = data
data = data1[:320]


def objective(params):
    learning_rate, hidden_size, attention_head_size, dropout, hidden_continuous_size, max_encoder_length, batch_size = params
    max_encoder_length = int(max_encoder_length)
    batch_size = int(batch_size)
    hidden_size=int(hidden_size)
    attention_head_size=int(attention_head_size)
    hidden_continuous_size=int(hidden_continuous_size)
    
    max_prediction_length = 1
    training_cutoff = data["time_idx"].max() - max_prediction_length
    
   
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="ChinaDADOU",
        group_ids=["id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["id"],
        time_varying_known_reals=["Month", "Year", "time_idx"],
        time_varying_unknown_reals=["ChinaDADOU","trading","gpr","wti","US",
                                    "Min of International price","Max of International price","Mean of International price","Max of WTI",
                                    "S1","S2","S3","S4","S5","S6","S7","S8"],
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    test_data = data[lambda x: x.time_idx > x.time_idx.max() - int(max_encoder_length)]
    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=15,
        gpus=0,
        weights_summary="top",
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[LearningRateMonitor(), EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=False, mode="min")],
        logger=TensorBoardLogger("lightning_logs")
    )
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    raw_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)
    tft.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True)

    #print(raw_predictions)
    new_raw_predictions, new_x =  tft.predict(
        test_data,
        mode="raw",
        return_x=True
    )

    tft.plot_prediction(
        new_x,
        new_raw_predictions,
        idx=0,
        show_future_observed=False
    )

    result=[]
    len_data=len(data1)
    test_len=100
    test_y=[]
    for i in range(test_len):
        test_data=data1[len_data-max_encoder_length-(test_len-i)+1:len_data-(test_len-i)+1]
        new_raw_predictions, new_x =  tft.predict(
        test_data,
        mode="raw",
        return_x=True
        )
        a=new_raw_predictions['prediction']
        b=a.numpy()
        result.append(b[0,0,3])

    test_y1=data1["ChinaDADOU"]
    test_y2=test_y1[len_data-test_len:len_data]
    test_y=[]
    for i in range(test_len):
        test_y.append(test_y2[[len_data+i-test_len]])
    result=np.array(result)
    test_y=np.array(test_y)
    MAPE1 = 0
    for i in range(test_len):
        MAPE1=MAPE1+np.abs(result[i]-test_y[i])/test_y[i]
    MAPE1 = MAPE1/test_len
    print(f"MAPE: {MAPE1}")
    return MAPE1
 

def sao(N, Max_iter, lb, ub, dim, fobj):
    if np.isscalar(ub):
        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

    X = initialization_sao(N, dim, ub, lb)
    Best_pos = np.zeros(dim)
    Best_score = np.inf
    Objective_values = np.zeros(N)

    Convergence_curve = []
    N1 = N // 2
    Elite_pool = []

    for i in range(N):
        Objective_values[i] = fobj(X[i, :])
        if i == 0:
            Best_pos = X[i, :].copy()
            Best_score = Objective_values[i]
        elif Objective_values[i] < Best_score:
            Best_pos = X[i, :].copy()
            Best_score = Objective_values[i]

    idx1 = np.argsort(Objective_values)
    second_best = X[idx1[1], :].copy()
    third_best = X[idx1[2], :].copy()
    half_best_mean = np.mean(X[idx1[:N1], :], axis=0)
    Elite_pool.append(Best_pos)
    Elite_pool.append(second_best)
    Elite_pool.append(third_best)
    Elite_pool.append(half_best_mean)

    Convergence_curve.append(Best_score)

    Na = N // 2
    Nb = N // 2

    l = 1
    while l <= Max_iter:
        RB = np.random.randn(N, dim)
        T = np.exp(-l / Max_iter)
        k = 1
        DDF = 0.35 * (1 + (5/7) * ((np.exp(l / Max_iter) - 1) ** k) / ((np.exp(1) - 1) ** k))
        M = DDF * T

        X_centroid = np.mean(X, axis=0)

        index1 = np.random.choice(N, Na, replace=False)
        index2 = np.setdiff1d(np.arange(N), index1)

        for i in index1:
            r1 = np.random.rand()
            k1 = np.random.randint(0, 4)
            for j in range(dim):
                X[i, j] = Elite_pool[k1][j] + RB[i, j] * (r1 * (Best_pos[j] - X[i, j]) + (1 - r1) * (X_centroid[j] - X[i, j]))

        if Na < N:
            Na += 1
            Nb -= 1

        if Nb >= 1:
            for i in index2:
                r2 = 2 * np.random.rand() - 1
                for j in range(dim):
                    X[i, j] = M * Best_pos[j] + RB[i, j] * (r2 * (Best_pos[j] - X[i, j]) + (1 - r2) * (X_centroid[j] - X[i, j]))

        for i in range(N):
            X[i, :] = np.clip(X[i, :], lb, ub)
            Objective_values[i] = fobj(X[i, :])
            if Objective_values[i] < Best_score:
                Best_pos = X[i, :].copy()
                Best_score = Objective_values[i]

        idx1 = np.argsort(Objective_values)
        second_best = X[idx1[1], :].copy()
        third_best = X[idx1[2], :].copy()
        half_best_mean = np.mean(X[idx1[:N1], :], axis=0)
        Elite_pool[0] = Best_pos
        Elite_pool[1] = second_best
        Elite_pool[2] = third_best
        Elite_pool[3] = half_best_mean

        Convergence_curve.append(Best_score)
        l += 1

    return Best_pos, Best_score, Convergence_curve

def initialization_sao(SearchAgents_no, dim, ub, lb):
    boundary_diff = ub - lb
    Positions = np.random.rand(SearchAgents_no, dim) * boundary_diff + lb
    return Positions


SearchAgents_no = 10  #30
Max_iteration = 4 # 
dim = 7
lb = np.array([0.001, 6, 1, 0.1, 2, 2, 10])
ub = np.array([0.1, 12, 4, 0.5, 6, 10, 100]) #


best_pos, best_score, sao_curve = sao(SearchAgents_no, Max_iteration, lb, ub, dim, objective)

print(f"Best Loss: {best_score}")
print(f"Best Params: {best_pos}")
