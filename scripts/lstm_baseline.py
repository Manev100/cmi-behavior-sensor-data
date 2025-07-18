
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
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupShuffleSplit, GroupKFold, ParameterGrid
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
    # Missing values
    # Remove sequences without thm and tof
    no_nans = data.drop_nulls()
    with_nans = data.join(no_nans, on=["subject", "sequence_id", "row_id"], how='anti')
    n_nulls = with_nans.with_columns(pl.sum_horizontal(pl.all().is_null()).alias("Nulls")).sort(by="Nulls")
    sequences1 = n_nulls.filter(pl.col("Nulls") == 325).select(["sequence_id"]).unique().to_series().implode()
    missing_thm_tof_seqs = data.filter(pl.col("sequence_id").is_in(sequences1) )

    # Some sequences with thm_5 and all tof_5, and some sequences with thm_1, thm_5 and all tof_5 missing for whole sequence
    # impute with 0
    sequences2 = n_nulls.filter((pl.col("Nulls") == 65) | (pl.col("Nulls") == 66)).select(["sequence_id"]).unique().to_series().implode()
    impute_seqs = (data
                .filter(pl.col("sequence_id").is_in(sequences2))
                .with_columns(pl.all().fill_null(0)))

    # Some sequences with rot missing
    # impute with null or omit?
    sequences4 = n_nulls.filter((pl.col("Nulls") == 4) ).select(["sequence_id"]).unique().to_series().implode()
    rot_fill_seqs = (data
                .filter(pl.col("sequence_id").is_in(sequences4))
                .with_columns(pl.all().fill_null(0)))


    # Some sequences with single rot or thm values missing
    # backfill/forwardfill if possible
    sequences3 = n_nulls.filter((pl.col("Nulls") == 1)).select(["sequence_id"]).unique().to_series().implode()
    backfill_seqs = (data
                .filter(pl.col("sequence_id").is_in(sequences3))
                .with_columns(pl.all().fill_null(strategy="forward").over("sequence_id"))
                .with_columns(pl.all().fill_null(strategy="backward").over("sequence_id"))
                .with_columns(pl.all().fill_null(0)))

    other_seqs = data.filter(~pl.col("sequence_id").is_in(sequences1) & 
                                ~pl.col("sequence_id").is_in(sequences2) & 
                                ~pl.col("sequence_id").is_in(sequences3) & 
                                ~pl.col("sequence_id").is_in(sequences4))

    # missing_thm_tof_seqs are omitted
    # Remove sequence with missing Gesture phase
    data_prep = (pl.concat([other_seqs, impute_seqs, rot_fill_seqs, backfill_seqs], how="vertical").sort(by="row_id")
                    .filter(pl.col("sequence_id") != 'SEQ_011975'))

    assert len(data_prep.select("row_id").unique()) == len(data_prep), "Duplicate rows!"
    assert data_prep.null_count().sum_horizontal().item() == 0, "Missing_values!"
    return {'train': data_prep, 'val': None, 'test': None}

# data: pl. Dataframe, vanilla competition data
# features: List of features
def clean_dataset_quick(data, feature):
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
    
    # data_dict["train"] = pl.from_pandas(ct.fit_transform(data_dict["train"].to_pandas()))
    
    # for part in ["val", "test"]:
    #     if data_dict[part] is not None:
    #         data_dict[part] = ct.transform(data_dict[part])
    
    return data_dict, scaling_dict


# %%
# train, val, test: Split and preprocessed dataframes
# features_dict: Maps features type to list of features (e.g. acc: ["acc_x", "acc_y", "acc_z"])
def create_dataset(train, val, test, features_dict):
    
    def perpare_part(part, tail_length=75):
        sequences = {col_name: [] for col_name in list(features_dict.keys()) + [ "target"]}
        for name, data in (part
                            .sort(by=['sequence_id', 'sequence_counter'])
                            .group_by("sequence_id")
                            .tail(tail_length)
                            .group_by("sequence_id")
                            ):
            
            
            
            for col_name, cols in features_dict.items():
                array = data.select(cols).to_numpy()
                if array.shape[0] < tail_length:
                    padding = np.zeros((tail_length -  array.shape[0], array.shape[1]) , dtype=float)
                    array = np.vstack((padding, array))
                    
                sequences[col_name].append(array)
            
            sequences["target"].append(data.select("gesture_id").tail(1).item())
            
        return sequences

    data = {}
    for part_name, part in zip(["train", "val", "test"], [train, val, test]):
        if part is not None:
            part_data = perpare_part(part)
            data[part_name] = {'x_acc': np.array(part_data["acc"]).astype(np.float32), 
                               'x_rot': np.array(part_data["rot"]).astype(np.float32), 
                               'x_thm': np.array(part_data["thm"]).astype(np.float32), 
                               'x_tof': np.array(part_data["tof"]).astype(np.float32), 
                               'y': np.array(part_data["target"]),
                               }
            
    
    return data


# _data: pl. Dataframe, vanilla test data
# _ct: dictionary mapping columns to namedtuple of (mean, std) for preprocessing
# features_dict: Maps features type to list of features (e.g. acc: ["acc_x", "acc_y", "acc_z"])
def create_test_dataset(_data, _ct, features_dict):
    def clean_single(_data):
        return (_data
                .with_columns(pl.all().fill_null(strategy="forward").over("sequence_id"))
                .with_columns(pl.all().fill_null(strategy="backward").over("sequence_id"))
                .with_columns(pl.all().fill_null(0))
                .with_columns(pl.lit(0).alias("gesture_id"))
                .with_columns([pl.lit(0).alias(col) for col in ['phase', 'sequence_type', 'gesture', 'behavior', 'orientation']])
                )
    
    def preprocess_test(_data):
        return _data.with_columns([(pl.col(col)-_ct[col].mean) / _ct[col].std  for cols in features_dict.values() for col in cols])
         
    
    def perpare_part(part, tail_length=75):
        sequences = {col_name: [] for col_name in list(features_dict.keys()) + [ "target"]}
        for name, data in (part
                            .sort(by=['sequence_id', 'sequence_counter'])
                            .group_by("sequence_id")
                            .tail(tail_length)
                            .group_by("sequence_id")
                            ):
            
            for col_name, cols in features_dict.items():
                array = data.select(cols).to_numpy()
                if array.shape[0] < tail_length:
                    padding = np.zeros((tail_length -  array.shape[0], array.shape[1]) , dtype=float)
                    array = np.vstack((padding, array))
                    
                sequences[col_name].append(array)
            
            sequences["target"].append(data.select("gesture_id").tail(1).item())
            
        return sequences

    tmp = clean_single(_data)
    tmp = preprocess_test(tmp)
    
    part_name = "test"
    
    data = {}
    part_data = perpare_part(tmp)
    data[part_name] = {'x_acc': np.array(part_data["acc"]).astype(np.float32), 
                        'x_rot': np.array(part_data["rot"]).astype(np.float32), 
                        'x_thm': np.array(part_data["thm"]).astype(np.float32), 
                        'x_tof': np.array(part_data["tof"]).astype(np.float32), 
                        'y': np.array(part_data["target"]),
                        }
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
    data_torch = {
        part: {'X': torch.as_tensor((np.dstack([(data[part][mtype]) for mtype in ["x_acc", "x_rot", "x_thm", "x_tof"]])), device=device) }
        for part in data
    }
            
    return data_torch
    




class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, classes_dim, num_layers, dropout_rate=0):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout_rate)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2class = nn.Linear(hidden_dim, classes_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        class_space = self.hidden2class(lstm_out[-1,...])
        # class_scores = F.softmax(class_space, dim=1)
        return class_space
    


def prepare_model(data_prep, config):
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Convert data to tensors
    # stack it together for now
    data = {
        part: {'X': torch.as_tensor((np.dstack([(data_prep[part][mtype]) for mtype in ["x_acc", "x_rot", "x_thm", "x_tof"]])), device=device) }
        for part in data_prep
    }

    for part in data_prep:
        data[part]['y'] = torch.as_tensor(data_prep[part]["y"], device=device) 


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
    model = LSTMClassifier(config["n_features"], config["hidden_size"], config["n_classes"], config["lstm_layers"], config["dropout"]).to(device)
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
                  "target": config["target"],
                  "le": config["le"]
                  }
        
    return model_dict, data





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
    le = model_dict["le"]
    
    
    
    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                torch.transpose(data[part]['X'][idx],0,1)
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
                torch.transpose(data[part]['X'][idx],0,1)
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
    def __init__(self, train_ds, train_demo_ds, verbose=False):
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
        
        self.le = LabelEncoder()
        self.le.fit(gestures)
        train_ds = train_ds.with_columns(pl.Series(name="gesture_id", values=self.le.transform(train_ds.select("gesture").to_series())))

        self.CONFIG = {
            "compile_model": False,
            
            "n_features": 332,
            "n_classes": len(self.le.classes_),
            "target": "gesture_id",
            "le": self.le
        }

        self.hyper_params = {
            "lstm_layers": 2,
            "hidden_size": 256,
            'dropout': 0.5,
            "learning_rate": 0.001, 
            "weight_decay": 3e-4,
            "n_epochs": 23,
            "patience": 0
        }
        if self.verbose:
            print("Cleaning...")
            
        # train_ds_prep =  clean_dataset(train_ds, self.features)
        train_ds_prep =  clean_dataset_quick(train_ds, self.features)
        if self.verbose:
            print("Preprocessing...")
        data_dict, self.ct = preprocess(train_ds_prep, self.features)
        if self.verbose:
            print("Preparing dataset...")
        data_prep = create_dataset(data_dict["train"], None, None, features_dict=self.features_dict)
        if self.verbose:
            print("Preparing model...")
        self.model_dict, self.data = prepare_model(data_prep, config=self.CONFIG | self.hyper_params)
        
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
        test_data = create_test_dataset(test_ds, _ct=self.ct, features_dict=self.features_dict)
        if self.verbose:
            print("Inference on test data...")
        preds = inference(self.model_dict, test_data, self.CONFIG | self.hyper_params, verbose=False)
        return self.le.inverse_transform(preds.argmax(1))



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

    test_ds = pl.read_csv(TEST_PATH)
    test_demo_ds = pl.read_csv(TEST_DEMO_PATH)

    lstm = LSTM_Predictor(train_ds, train_demo_ds)
    lstm.train_model("models/2025_06_27_lstm_baseline.pth")
    lstm.load_model("models/2025_06_27_lstm_baseline.pth")
    print(lstm.predict_test(test_ds, test_demo_ds))


