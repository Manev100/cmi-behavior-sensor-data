
# %%
import sys, os, glob
import gc
import time, math, random
import ast
from collections import namedtuple

import numpy as np
import pandas as pd
import pandas.api.types
import polars as pl
import polars.selectors as cs

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
from torch import Tensor

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, ParameterGrid
from sklearn import metrics
from sklearn.metrics import f1_score
import joblib

from tqdm.auto import tqdm

from cmi_2025 import score

# %%
seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

Stats = namedtuple('Stats', ['mean', 'std'])

# %%

# data: pl. Dataframe, vanilla competition data
# features: List of features
def clean_dataset(data, features):
    data_prep = (data
                .with_columns(pl.all().fill_null(strategy="forward").over("sequence_id"))
                .with_columns(pl.all().fill_null(strategy="backward").over("sequence_id"))
                .with_columns(pl.all().fill_null(0))
                .filter(pl.col("sequence_id") != 'SEQ_011975')
                )
    return {'train': data_prep, 'val': None, 'test': None}


# %%
# data_dict: dictionary with keys ['train', 'val', 'test'] and pl.Dataframe values. Split dataframe 
# features: List of features
def preprocess(data_dict, features):
    ct = ColumnTransformer(
            [('std', StandardScaler(), features)],
            verbose_feature_names_out=False, 
            remainder="passthrough"
        )
    ct.set_output(transform="pandas")

    means = data_dict["train"].select(pl.col(features)).mean().to_dicts()[0]
    std = data_dict["train"].select(pl.col(features)).std().to_dicts()[0]
    scaling_dict = {feat: Stats(means[feat], std[feat])   for feat in features}
    data_dict["train"] = data_dict["train"].with_columns([(pl.col(col)-scaling_dict[col].mean) / scaling_dict[col].std  for col in features])
    
    return data_dict, scaling_dict


# %%
# train, val, test: Split and preprocessed dataframes
# features_dict: Maps features type to list of features (e.g. acc: ["acc_x", "acc_y", "acc_z"])
def create_dataset(_data, _data_demo, features_dict, tail_length=75):
    
    def perpare_part(part, part_demo, tail_length=75):
        sequences = {col_name: [] for col_name in ["acc", "rot", "thm", "tof", "target", "subject", "demo"]}
        for name, data in (part
                            .sort(by=['sequence_id', 'sequence_counter'])
                            .group_by("sequence_id")
                            .tail(tail_length)
                            .group_by("sequence_id")
                            ):
            
            for col_name in ["acc", "rot", "thm", "tof"]:
                array = data.select(features_dict[col_name]).to_numpy()
                if array.shape[0] < tail_length:
                    padding = np.zeros((tail_length -  array.shape[0], array.shape[1]) , dtype=float)
                    array = np.vstack((padding, array))
                    
                sequences[col_name].append(array)
            
            sequences["target"].append(data.select("gesture_id").tail(1).item())
            subject = data.select("subject").tail(1).item()
            
            sequences["subject"].append(subject)
            sequences["demo"].append(part_demo.filter(pl.col("subject") == subject).select(features_dict["demo"]).to_numpy())
            
        return sequences

   
    data = {}
    
    part_data = perpare_part(_data, _data_demo, tail_length=tail_length)
    data['train'] = {'x_acc': np.array(part_data["acc"]).astype(np.float32), 
                    'x_rot': np.array(part_data["rot"]).astype(np.float32), 
                    'x_thm': np.array(part_data["thm"]).astype(np.float32), 
                    'x_tof': np.array(part_data["tof"]).astype(np.float32), 
                    'demo': np.array(part_data["demo"]).astype(np.float32),
                    'y': np.array(part_data["target"]),
                    }
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # stack it together for now
    data_torch = {
            part: {'X': torch.as_tensor((np.dstack([(data[part][mtype]) for mtype in ["x_acc", "x_rot", "x_thm", "x_tof"]])), device=device),
                    'X_demo': torch.as_tensor(data[part]["demo"], device=device),
                    'y': torch.as_tensor(data[part]["y"], device=device) }
            for part in data
    }
    
    return data_torch


# _data: pl. Dataframe, vanilla test data
# _ct: dictionary mapping columns to namedtuple of (mean, std) for preprocessing
# features_dict: Maps features type to list of features (e.g. acc: ["acc_x", "acc_y", "acc_z"])
def create_test_dataset(data, data_demo, ct, ct_demo, features_dict, tail_length=100):
    def clean_single(_data):
        return (_data
                .with_columns(pl.all().fill_null(strategy="forward").over("sequence_id"))
                .with_columns(pl.all().fill_null(strategy="backward").over("sequence_id"))
                .with_columns(pl.all().fill_null(0))
                .with_columns([pl.lit(0).alias(col) for col in ['phase', 'sequence_type', 'gesture', 'behavior', 'orientation']])
                )
    
    def preprocess_test(_data, _ct, _features):
        return _data.with_columns([(pl.col(col)-_ct[col].mean) / _ct[col].std  for col in _features])
         
    
    def perpare_part(part, part_demo, tail_length=100):
        sequences = {col_name: [] for col_name in ["acc", "rot", "thm", "tof", "target", "subject", "demo"]}
        for name, data in (part
                            .sort(by=['sequence_id', 'sequence_counter'])
                            .group_by("sequence_id")
                            .tail(tail_length)
                            .group_by("sequence_id")
                            ):
            
            
            
            for col_name in ["acc", "rot", "thm", "tof"]:
                array = data.select(features_dict[col_name]).to_numpy()
                if array.shape[0] < tail_length:
                    padding = np.zeros((tail_length -  array.shape[0], array.shape[1]) , dtype=float)
                    array = np.vstack((padding, array))
                    
                sequences[col_name].append(array)
            
            if "gesture_id" not in data.columns:
                sequences["target"].append(0)
            else: 
                sequences["target"].append(data.select("gesture_id").tail(1).item())
                
            subject = data.select("subject").tail(1).item()
            
            sequences["subject"].append(subject)
            sequences["demo"].append(part_demo.filter(pl.col("subject") == subject).select(features_dict["demo"]).to_numpy())
            
        return sequences

    tmp = clean_single(data)
    tmp = preprocess_test(tmp, _ct=ct, _features=[col  for feature_set in ["acc", "rot", "thm", "tof"] for col in features_dict[feature_set] ])
    
    tmp_demo = preprocess_test(data_demo, _ct=ct_demo, _features=features_dict["demo"])
    
    part_name = "test"
    
    data = {}
    part_data = perpare_part(tmp, tmp_demo, tail_length=tail_length)
    data[part_name] = {'x_acc': np.array(part_data["acc"]).astype(np.float32), 
                        'x_rot': np.array(part_data["rot"]).astype(np.float32), 
                        'x_thm': np.array(part_data["thm"]).astype(np.float32), 
                        'x_tof': np.array(part_data["tof"]).astype(np.float32), 
                        'demo': np.array(part_data["demo"]).astype(np.float32),
                        'y': np.array(part_data["target"]),
                        }
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
    data_torch = {
        part: {'X': torch.as_tensor((np.dstack([(data[part][mtype]) for mtype in ["x_acc", "x_rot", "x_thm", "x_tof"]])), device=device) ,
               'X_demo': torch.as_tensor(data[part]["demo"], device=device),
               'y': torch.as_tensor(data[part]["y"], device=device)
               }
        
        for part in data
    }
            
    return data_torch
    



class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, input_demo_dim, hidden_dim, classes_dim, num_layers, dropout_rate=0):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        input_dim_rnn = 12+input_demo_dim
        output_dim_rnn = 12

        self.rnn = nn.RNN(input_dim_rnn, output_dim_rnn, num_layers=1, batch_first=True)
        
        # conv layer 1: K: 64x5, b: 64
        self.features_maps = [64, 64, 64, 64]
        self.kernel_sizes = [4,4,4,4]
        # self.conv1 = nn.Conv1d(input_dim, input_dim*self.features_maps[0], self.kernel_sizes[0], stride=1, groups=input_dim)
        self.conv1 = nn.Conv1d(1, self.features_maps[0], self.kernel_sizes[0], stride=1, groups=1)
        
        # conv layer 2-4:  K: 64x64x5, b: 64
        self.conv2 = nn.Conv2d(1, self.features_maps[1], (self.features_maps[0], self.kernel_sizes[1]), stride=1, groups=1)
        self.conv3 = nn.Conv2d(1, self.features_maps[2], (self.features_maps[1], self.kernel_sizes[1]), stride=1, groups=1)
        self.conv4 = nn.Conv2d(1, self.features_maps[3], (self.features_maps[2], self.kernel_sizes[1]), stride=1, groups=1)
        
        # tof conv layers:
        self.tof_features_maps = [8,16,32]
        self.tof_conv1 = nn.Conv3d(1, self.tof_features_maps[0], (5,3,3), stride=1)
        self.tof_conv2 = nn.Conv3d(self.tof_features_maps[0], self.tof_features_maps[1], (5,3,3), stride=1)
        self.tof_conv3 = nn.Conv3d(self.tof_features_maps[1], self.tof_features_maps[2], (5,3,3), stride=1)
        self.tof_maxp = nn.MaxPool3d(kernel_size=(1,2,2))
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(12*self.features_maps[3] + 5*self.tof_features_maps[2], hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2class = nn.Linear(hidden_dim+input_demo_dim, classes_dim)

    # x: ( BATCH_SIZE, time, features)
    # x_demo: (BATCH_SIZE, 1, demo features)
    def forward(self, x, x_demo):

        # 1.1 only acc, rot and thm (body heat) through rnn
        
        # demo input to (time, batch_size, demo features) and repeat in time dimension
        x_demo_repeated = x_demo.repeat(1, x.shape[1], 1)
        x_in = torch.cat([x[:, :, :12], x_demo_repeated], dim=2)
        rnn_out, _ = self.rnn(x_in)
        
        # swap channels and time dimension
        # (BATCH_SIZE, channels, time)
        rnn_out = rnn_out.transpose(1,2)
        # channels to batch dimensions to apply same convolution per channel
        # (BATCH_SIZE * channels, time)
        rnn_out = rnn_out.view(-1, 1, rnn_out.shape[-1])
        
        
        # 1.2 rnn output through CNN
        x_conv = nn.functional.relu(self.conv1(rnn_out))
        
        for i, conv in enumerate([self.conv2, self.conv3, self.conv4]):
            # print(f"Before conv {i+2} ", x_conv.shape)
            x_conv = nn.functional.relu(conv(x_conv.squeeze().unsqueeze(1)))
            # print(f"After conv {i+2} ", x_conv.shape)
        
        # After convolutions we get tensor of shape (BATCH_SIZE * channels, last-feature-map, 1, time)
        x_conv = x_conv.squeeze(2)
        # separate channels from batch dimenison
        x_conv = x_conv.view(-1, 12, x_conv.shape[1], x_conv.shape[2])
        # merge feature map dimension into channels dimension 
        x_conv = x_conv.view(x_conv.shape[0],-1, x_conv.shape[3])
        # Transformed tensor to shape (BATCH_SIZE, channels * last-feature-map, time)
        # -> 64*12=768 Features
        
        # 2.1 tof features
        x_tof = x[:, :, 12:].view(x.shape[0],x.shape[1], 5, 8, 8)
        
        # 2.2. tof Features through CNN
        # (BATCH_SIZE, time, Depth, H, W) -> (BATCH_SIZE, Depth, time, H, W)
        x_tof = x_tof.transpose(1,2)
        # (BATCH_SIZE, Depth, time, H, W) -> (BATCH_SIZE * Depth, time, H, W)
        x_tof = x_tof.reshape(-1,1, x_tof.shape[2], x_tof.shape[3], x_tof.shape[4])
        
        # 8x8 -> 4x4x8 -> 2x2x16 -> 1x1x32 -> 32*5=160 features
        x_tof = nn.functional.relu(self.tof_conv1(x_tof))
        x_tof = nn.functional.relu(self.tof_conv2(x_tof))
        x_tof = nn.functional.relu(self.tof_maxp(self.tof_conv3(x_tof)))
        
        # Separate Batchsize from Depth dimension
        x_tof = x_tof.squeeze()
        # Separate batch and channels (BATCH_SIZE, Channels, feature maps, time)
        x_tof = x_tof.view(-1, 5, x_tof.shape[1], x_tof.shape[2])
        # Flatten channel and feature maps dim (BATCH_SIZE, Channels*feature maps, time)
        x_tof = x_tof.view(x_tof.shape[0],-1, x_tof.shape[3])
        
        # Cat acc, rot, thm and tof feature maps
        lstm_in = torch.cat([x_conv, x_tof], dim=1)
        lstm_out, _ = self.lstm(lstm_in.transpose(1,2))

        x_cat = torch.cat([lstm_out[:,-1,:], x_demo.squeeze(1)], 1)
        class_space = self.hidden2class(x_cat)
        class_scores = F.softmax(class_space, dim=1)
        return class_scores
    


class LSTMClassifierIMUonly(nn.Module):

    def __init__(self, input_dim, input_demo_dim, hidden_dim, classes_dim, num_layers, dropout_rate=0):
        super(LSTMClassifierIMUonly, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        input_dim_rnn = 7+input_demo_dim
        output_dim_rnn = 7

        self.rnn = nn.RNN(input_dim_rnn, output_dim_rnn, num_layers=1, batch_first=True)
        
        # conv layer 1: K: 64x5, b: 64
        self.features_maps = [64, 64, 64, 64]
        self.kernel_sizes = [4,4,4,4]
        # self.conv1 = nn.Conv1d(input_dim, input_dim*self.features_maps[0], self.kernel_sizes[0], stride=1, groups=input_dim)
        self.conv1 = nn.Conv1d(1, self.features_maps[0], self.kernel_sizes[0], stride=1, groups=1)
        
        # conv layer 2-4:  K: 64x64x5, b: 64
        self.conv2 = nn.Conv2d(1, self.features_maps[1], (self.features_maps[0], self.kernel_sizes[1]), stride=1, groups=1)
        self.conv3 = nn.Conv2d(1, self.features_maps[2], (self.features_maps[1], self.kernel_sizes[1]), stride=1, groups=1)
        self.conv4 = nn.Conv2d(1, self.features_maps[3], (self.features_maps[2], self.kernel_sizes[1]), stride=1, groups=1)
             
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(7*self.features_maps[3] , hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2class = nn.Linear(hidden_dim+input_demo_dim, classes_dim)

    # x: ( BATCH_SIZE, time, features)
    # x_demo: (BATCH_SIZE, 1, demo features)
    def forward(self, x, x_demo):

        # 1.1 only acc, rot and thm (body heat) through rnn
        
        # demo input to (time, batch_size, demo features) and repeat in time dimension
        x_demo_repeated = x_demo.repeat(1, x.shape[1], 1)
        x_in = torch.cat([x[:, :, :7], x_demo_repeated], dim=2)
        rnn_out, _ = self.rnn(x_in)
        
        # swap channels and time dimension
        # (BATCH_SIZE, channels, time)
        rnn_out = rnn_out.transpose(1,2)
        # channels to batch dimensions to apply same convolution per channel
        # (BATCH_SIZE * channels, time)
        rnn_out = rnn_out.view(-1, 1, rnn_out.shape[-1])
        
        
        # 1.2 rnn output through CNN
        x_conv = nn.functional.relu(self.conv1(rnn_out))
        
        for i, conv in enumerate([self.conv2, self.conv3, self.conv4]):
            # print(f"Before conv {i+2} ", x_conv.shape)
            x_conv = nn.functional.relu(conv(x_conv.squeeze().unsqueeze(1)))
            # print(f"After conv {i+2} ", x_conv.shape)
        
        # After convolutions we get tensor of shape (BATCH_SIZE * channels, last-feature-map, 1, time)
        x_conv = x_conv.squeeze(2)
        # separate channels from batch dimenison
        x_conv = x_conv.view(-1, 7, x_conv.shape[1], x_conv.shape[2])
        # merge feature map dimension into channels dimension 
        x_conv = x_conv.view(x_conv.shape[0],-1, x_conv.shape[3])
        # Transformed tensor to shape (BATCH_SIZE, channels * last-feature-map, time)
        # -> 64*12=768 Features
                
        # Cat acc, rot, thm and tof feature maps
        lstm_out, _ = self.lstm(x_conv.transpose(1,2))

        x_cat = torch.cat([lstm_out[:,-1,:], x_demo.squeeze(1)], 1)
        class_space = self.hidden2class(x_cat)
        class_scores = F.softmax(class_space, dim=1)
        return class_scores



def prepare_model(config):
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Automatic mixed precision (AMP)
    # torch.float16 is implemented for completeness,
    # but it was not tested in the project,
    # so torch.bfloat16 is used by default.
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )
    # Changing False to True will result in faster training on compatible hardware.
    amp_enabled = False and amp_dtype is not None
    grad_scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None  # type: ignore

    # torch.compile
    compile_model = config["compile_model"]

    # fmt: off
    print(
        f'Device:        {device.type.upper()}'
        f'\nAMP:           {amp_enabled} (dtype: {amp_dtype})'
        f'\ntorch.compile: {compile_model}'
    )
    
    # Choose one of the two configurations below.
    # TODO
    if config["imu_only"]:
        model = LSTMClassifierIMUonly(config["n_features"], config["n_demo_features"], config["hidden_size"], config["n_classes"], config["lstm_layers"], config["dropout"]).to(device)
    else:
        model = LSTMClassifier(config["n_features"], config["n_demo_features"], config["hidden_size"], config["n_classes"], config["lstm_layers"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    if compile_model:
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument
        # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode
        
    model_dict = {"model": model,
                  "eval_mode": evaluation_mode,
                  "optimizer": optimizer,
                  "device": device,
                  "grad_scaler": grad_scaler,
                  "amp_enabled": amp_enabled,
                  "amp_dtype": amp_dtype,
                  "target": config["target"]
                  }
        
    return model_dict





# %%


def train(model_dict, data, config, verbose=False):
    model = model_dict["model"]
    optimizer = model_dict["optimizer"]
    evaluation_mode = model_dict["eval_mode"]
    device = model_dict["device"]
    grad_scaler = model_dict["grad_scaler"]
    amp_enabled = model_dict["amp_enabled"]
    amp_dtype = model_dict["amp_dtype"]
    target = model_dict["target"]
    le = config["le"]
    
    
    
    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['X'][idx],
                data[part]['X_demo'][idx]
            )
        )

    task_type = "classification"
    base_loss_fn = F.mse_loss if task_type == 'regression' else F.cross_entropy


    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return base_loss_fn(y_pred, y_true)

    def score_fn(y_true, y_pred):
        sol = pd.DataFrame({"gesture": le.inverse_transform(y_true)}).reset_index(names=["id"])
        sub = pd.DataFrame({"gesture": le.inverse_transform(y_pred)}).reset_index(names=["id"])
        return score(sol, sub, row_id_column_name='id')

    @evaluation_mode()
    def evaluate(part: str) -> tuple[float, float]:
        model.eval()

        # When using torch.compile, you may need to reduce the evaluation batch size.
        eval_batch_size = 8096
        y_pred = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
        )


        loss = loss_fn(y_pred, data[part]["y"]).detach().cpu().numpy()

        if task_type != 'regression':
            # For classification, the mean must be computed in the probabily space.
            y_pred = F.softmax(y_pred, dim=1).cpu().numpy()

        y_true = data[part]['y'].cpu().numpy()
        
        sc = (
            score_fn(y_true, y_pred.argmax(1))
        )
        return float(sc), float(loss)  # The higher -- the better.

    
    # For demonstration purposes (fast training and bad performance),
    # one can set smaller values:
    # n_epochs = 20
    # patience = 2
    n_epochs = 1_000_000_000
    if "n_epochs" in config:
        n_epochs =  config["n_epochs"]
    
    # Early stopping: the training stops when
    # there are more than `patience` consequtive bad updates.
    patience = 10
    if "patience" in config:
        patience =  config["patience"]
    

    batch_size = 256
    epoch_size = math.ceil(len(data["train"]["X"]) / batch_size)
    best = {
        'val': -math.inf,
        'test': -math.inf,
        'epoch': -1,
    }
    
    remaining_patience = patience

    if verbose:
        print('-' * 88 + '\n')
    
       
    
    for epoch in range(n_epochs):
        pred_train = torch.zeros((len(data["train"]["X"]), config["n_classes"]), device=device)
        for batch_idx in tqdm(
            torch.randperm(len(data['train']['y']), device=device).split(batch_size),
            desc=f'Epoch {epoch}',
            total=epoch_size,
            disable=not verbose
        ):
            model.train()
            optimizer.zero_grad()
            pred = apply_model('train', batch_idx)
            loss = loss_fn(pred, data["train"]["y"][batch_idx])
            pred_train[batch_idx] = pred.detach()
            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()  # type: ignore
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
        train_loss = loss_fn(pred_train, data["train"]["y"]).cpu().numpy()
        train_score = float(score_fn(data["train"]["y"].cpu().numpy(), 
                                        F.softmax(pred_train, dim=1).cpu().numpy().argmax(1)) )
               
    return train_loss, train_score





# %%


def inference(model_dict, data, config, verbose=False):
    model = model_dict["model"]
    optimizer = model_dict["optimizer"]
    evaluation_mode = model_dict["eval_mode"]
    device = model_dict["device"]
    grad_scaler = model_dict["grad_scaler"]
    amp_enabled = model_dict["amp_enabled"]
    amp_dtype = model_dict["amp_dtype"]
    target = model_dict["target"]
    
    
    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['X'][idx],
                data[part]['X_demo'][idx]
            )
        )

    @evaluation_mode()
    def evaluate(part: str):
        model.eval()

        eval_batch_size = 8096
        y_pred = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['X']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
        )

        y_pred = F.softmax(y_pred, dim=1).cpu().numpy()


        
        return y_pred  # The higher -- the better.

               
    return evaluate("test")


class LSTM_Predictor():
    def __init__(self, train_ds, train_demo_ds, imu_only=False, verbose=False):
        self.verbose = verbose
        train_cols = list(train_ds.columns)
        acc_cols = [col  for col in train_cols if col.startswith("acc")]
        rot_cols = [col  for col in train_cols if col.startswith("rot")]
        thm_cols = [col  for col in train_cols if col.startswith("thm")]
        tof_cols = [[col  for col in train_cols if col.startswith(f"tof_{i+1}")] for i in range(5)]
        tof_cols_all = [col for cl in tof_cols for col in cl]
        self.features_dict = {"acc":acc_cols, "rot":rot_cols, "thm":thm_cols, "tof":tof_cols_all}
        self.features = acc_cols+rot_cols+thm_cols+tof_cols_all
        gestures = ['Pull air toward your face', 'Feel around in tray and pull out an object', 'Neck - scratch', 'Pinch knee/leg skin', 
                'Forehead - scratch', 'Eyelash - pull hair', 'Drink from bottle/cup', 'Wave hello', 'Cheek - pinch skin', 
                'Forehead - pull hairline', 'Text on phone', 'Write name in air', 'Scratch knee/leg skin', 'Neck - pinch skin', 
                'Write name on leg', 'Above ear - pull hair', 'Eyebrow - pull hair', 'Glasses on/off']
        
        self.demo_features = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
        self.features_dict["demo"] = self.demo_features
        
        self.le = LabelEncoder()
        self.le.fit(gestures)
        train_ds = train_ds.with_columns(pl.Series(name="gesture_id", values=self.le.transform(train_ds.select("gesture").to_series())))

        self.CONFIG = {
            "compile_model": False,
            
            "n_features": 332,
            "n_demo_features": len(self.demo_features),
            "n_classes": len(self.le.classes_),
            "target": "gesture_id",
            "tail_length": 100,
            "le": self.le,
            "imu_only": imu_only
        }

        self.hyper_params = {
            "lstm_layers": 2,
            "hidden_size": 64,
            'dropout': 0.2,
            "learning_rate": 1e-3, 
            "weight_decay": 0.9,
            "n_epochs": 60,
            "patience": 84
        }
        if self.verbose:
            print("Cleaning...")
            
        # train_ds_prep =  clean_dataset(train_ds, self.features)
        train_ds_prep =  clean_dataset(train_ds, self.features)
        train_demo_ds_prep =  {'train': train_demo_ds}
        if self.verbose:
            print("Preprocessing...")
        data_dict, self.ct = preprocess(train_ds_prep, self.features)
        data_demo_dict, self.ct_demo = preprocess(train_demo_ds_prep, self.demo_features)
        if self.verbose:
            print("Preparing dataset...")
        self.data = create_dataset(data_dict["train"], data_demo_dict["train"], features_dict=self.features_dict, tail_length=self.CONFIG["tail_length"])
        if self.verbose:
            print("Preparing model...")
        self.model_dict = prepare_model(config=self.CONFIG | self.hyper_params)
        
    # PATH: Path to save model to, e.g. 'model.pth'
    def train_model(self, PATH=None):
        if self.verbose:
            print("Training model...")
        train(self.model_dict, self.data, self.CONFIG | self.hyper_params, verbose=False)
        
        if PATH is not None:
            if self.verbose:
                print("Saving model...")
            torch.save(self.model_dict["model"].state_dict(), PATH)
        
    # PATH: saved model path, e.g. 'model.pth'
    def load_model(self, PATH):
        if self.verbose:
            print("Loading model...")
            
        if self.model_dict["device"].type == 'cpu':  
            self.model_dict["model"].load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        else:
            self.model_dict["model"].load_state_dict(torch.load(PATH))
        
    def predict_test(self, test_ds, test_demo_ds):
        if self.verbose:
            print("Preparing test data...")
        test_data = create_test_dataset(test_ds, test_demo_ds, ct=self.ct, ct_demo=self.ct_demo, features_dict=self.features_dict, tail_length=self.CONFIG["tail_length"])
        if self.verbose:
            print("Inference on test data...")
        preds = inference(self.model_dict, test_data, self.CONFIG | self.hyper_params, verbose=False)
        return self.le.inverse_transform(preds.argmax(1))
    
    # labels=True means predictions and true labels are given as string labels instead of numbers
    def score_test(self, test_ds, test_demo_ds):
        if self.verbose:
            print("Preparing test data...")
        if "gesture" in test_ds.columns:
            test_ds = test_ds.with_columns(pl.Series(name="gesture_id", values=self.le.transform(test_ds.select("gesture").to_series())))       
            
        test_data = create_test_dataset(test_ds, test_demo_ds, ct=self.ct, ct_demo=self.ct_demo, features_dict=self.features_dict, tail_length=self.CONFIG["tail_length"])
        if self.verbose:
            print("Inference on test data...")
        preds = inference(self.model_dict, test_data, self.CONFIG | self.hyper_params, verbose=False)
        preds = self.le.inverse_transform(preds.argmax(1))
        
        true = self.le.inverse_transform(test_data["test"]['y'].cpu())
        
        sol = pd.DataFrame({"gesture": true}).reset_index(names=["id"])
        sub = pd.DataFrame({"gesture": preds}).reset_index(names=["id"])

        return score(sol, sub, row_id_column_name='id')


def split_data(data, data_demo):
    sequences = (data
                .group_by(["sequence_id", "subject"])
                .agg(pl.col("gesture").first())
                )
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    train_index, test_index = next(sgkf.split(sequences, 
                                            sequences.select("gesture").to_series(), 
                                            sequences.select("subject").to_series() ))
    train_index2, test_index2 = next(sgkf2.split(sequences[test_index], 
                                            sequences[test_index].select("gesture").to_series(), 
                                            sequences[test_index].select("subject").to_series() ))
    
    data_dict = {}
    
    for part_name, part_index in zip(["train", "val", "test"], [train_index, train_index2, test_index2]):
        data_dict[part_name] = data.filter(pl.col("sequence_id").is_in(sequences[part_index].select("sequence_id").to_series().implode()))
        data_dict[part_name + "_demo"] = data_demo.filter(pl.col("subject").is_in(sequences[part_index].select("subject").to_series().implode()))
        
    return data_dict

if __name__ == '__main__': 
        
    # %%
    BASE = "."
    COMP_DATA_BASE = os.path.join(BASE, "data", "raw")
    PREP_DATA_BASE = os.path.join(BASE, "data", "processed")
    FIGURES_BASE = os.path.join(BASE, "figures")

    TRAIN_PATH = os.path.join(COMP_DATA_BASE, "train.csv")
    TRAIN_DEMO_PATH = os.path.join(COMP_DATA_BASE, "train_demographics.csv")
    TEST_PATH = os.path.join(COMP_DATA_BASE, "test.csv")
    TEST_DEMO_PATH = os.path.join(COMP_DATA_BASE, "test_demographics.csv")

    train_ds = pl.read_csv(TRAIN_PATH)
    train_demo_ds = pl.read_csv(TRAIN_DEMO_PATH)

    data_dict = split_data(train_ds, train_demo_ds)
    
    
    test_ds = pl.read_csv(TEST_PATH)
    test_demo_ds = pl.read_csv(TEST_DEMO_PATH)

    lstm = LSTM_Predictor(data_dict["train"], data_dict["train_demo"], imu_only=True, verbose=True)
    lstm.train_model("models/2025_07_12_lstm_cnn.pth")
    lstm.load_model("models/2025_07_12_lstm_cnn.pth")
    
    score_train = lstm.score_test(data_dict["train"], data_dict["train_demo"])
    score_val = lstm.score_test(data_dict["val"], data_dict["val_demo"])
    score_test = lstm.score_test(data_dict["test"], data_dict["test_demo"])
    print(f"train score: {score_train:.4f}")
    print(f"val score: {score_val:.4f}")
    print(f"test score: {score_test:.4f}")
    
    print(lstm.predict_test(test_ds, test_demo_ds))


