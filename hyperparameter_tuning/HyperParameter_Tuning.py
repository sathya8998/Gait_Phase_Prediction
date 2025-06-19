import torch
import optuna
import torch.nn as nn
import logging
from Gait_Phase.evaluation_utils.evaluation_utils import F1Score
from Gait_Phase.gpu_utils import configure_gpu
from Gait_Phase.config import CONFIG
logger = logging.getLogger(__name__)

configure_gpu()

class EarlyStopping:
    def __init__(self, monitor='val_loss', patience=3, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best = float('inf')
        self.counter = 0

    def step(self, current):
        if current < self.best:
            self.best = current
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def hyperparameter_tuning(model_builder, tuner_type, X_train, y_train, X_val, y_val, input_shape, num_classes, fold,
                          strategy):
    try:
        # Set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Choose sampler and pruner based on tuner type
        if tuner_type == 'BayesianOptimization':
            sampler = optuna.samplers.TPESampler()
            pruner = None
        elif tuner_type == 'RandomSearch':
            sampler = optuna.samplers.RandomSampler()
            pruner = None
        elif tuner_type == 'Hyperband':
            sampler = optuna.samplers.TPESampler()
            pruner = optuna.pruners.HyperbandPruner()
        else:
            logger.error(f"Unsupported tuner type: {tuner_type}")
            return None, None

        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

        class OptunaHP:
            def __init__(self, trial):
                self.trial = trial

            def Int(self, name, min_value, max_value, step, default=None):
                values = list(range(min_value, max_value + 1, step))
                return self.trial.suggest_categorical(name, values)

            def Float(self, name, min_value, max_value, step=None, default=None, sampling=None):
                if sampling == 'log':
                    return self.trial.suggest_float(name, min_value, max_value, log=True)
                elif step is not None:
                    return self.trial.suggest_float(name, min_value, max_value, step=step)
                else:
                    return self.trial.suggest_float(name, min_value, max_value)

            def Boolean(self, name, default=False):
                return self.trial.suggest_categorical(name, [False, True])

            def Choice(self, name, values):
                return self.trial.suggest_categorical(name, values)

            # NEW: Add Categorical method so that model builders can call hp.Categorical.
            def Categorical(self, name, values):
                return self.trial.suggest_categorical(name, values)

        def objective(trial):
            hp = OptunaHP(trial)
            model, lr = model_builder.build_model_hp(hp, input_shape, num_classes)
            if model is None:
                raise ValueError("Model building failed; received None")
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            X_train_tensor = torch.tensor(X_train).float()
            y_train_tensor = torch.tensor(y_train).long()
            X_val_tensor = torch.tensor(X_val).float()
            y_val_tensor = torch.tensor(y_val).long()
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                drop_last=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=False,
                drop_last=True
            )

            epochs = CONFIG['epochs']
            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss_total += loss.item() * batch_x.size(0)
                val_loss = val_loss_total / len(val_dataset)

            # Compute F1 metric on the validation set.
            f1_metric = F1Score(num_classes=num_classes)
            f1_metric.reset_states()
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    f1_metric.update_state(batch_y, outputs)
            val_f1_score = f1_metric.result()

            # Clear the GPU cache to free any unallocated memory before the next trial.
            torch.cuda.empty_cache()

            return val_f1_score

        study.optimize(objective, n_trials=CONFIG['tuner_trials'])
        best_trial = study.best_trial
        logger.info(f"Best hyperparameters for {strategy} on fold {fold}: {best_trial.params}")
        return study, best_trial

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning with {tuner_type}: {e}")
        return None, None
