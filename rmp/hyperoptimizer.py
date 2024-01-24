import datetime
import logging
from pathlib import Path

import dill
import hyperopt
import petname
import torch.nn as nn
import wandb
from hyperopt import STATUS_OK, Trials, fmin, tpe
from torch.utils.data import Dataset

from rmp.dataloader import ModelPhases, RpmSignals
from rmp.global_config import RESULT_DIR
from rmp.gym import BaseGym, DeepGym, MachineGym
from rmp.models import (
    DecompLinear,
    LinearOffline,
    Many2Many,
    ModelArch,
    Transformer,
    TransformerTSFv2,
    XGBoostTSF,
    YourCustomModel,
)
from rmp.utils.common_types import PathLike
from rmp.utils.decorators import convert
from rmp.utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


def partial(func, /, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs, **newkeywords)

    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc


class Hyperoptimizer(LoggerMixin):
    def __init__(self, search_space: dict, constant_config: dict):
        self.search_space = search_space
        self.constant_config = constant_config
        self.start_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    def function_to_minimize(self, hyper_parameters: dict) -> dict:
        self.constant_config["training_phase"] = BaseGym.calc_number_training_steps(
            input_features=hyper_parameters["input_features"],
            output_features=self.constant_config["output_features"],
            training_phase_s=self.constant_config["training_phase_s"],
        )
        model = self.init_model(
            hyper_para=hyper_parameters, constant_config=self.constant_config
        )
        train_dataset, dev_dataset, _ = self.init_datasets(
            hyper_para=hyper_parameters, constant_config=self.constant_config
        )
        pred_horizon = (
            40 * self.constant_config["future_steps"]
        )  # in ms; SPS=25 Hz assumed
        use_wandb = False
        try:
            wandb.init(
                config={**self.constant_config, **hyper_parameters},
                project=f"TSF-{self.constant_config['model_arch'].value}",
                dir=RESULT_DIR,
                anonymous="allow",
            )
            run_name = wandb.run.name
            use_wandb = True
        except wandb.errors.Error:
            run_name = petname.generate(3)
            logger.warning(
                "Cannot import weights and biases. Consider creating a free account and login for advanced results tracking!"
            )
        result_dir = (
            RESULT_DIR
            / "HYPEROPT2"
            / f"{self.constant_config['model_arch'].value}_{self.start_datetime}"
            / f"{run_name}_{pred_horizon}_ms_output_{self.constant_config['output_features']}"
        )
        result_dir.mkdir(exist_ok=True, parents=True)
        logger.info(
            f"Local result directory of current hyper-optimizer run: {result_dir}"
        )

        if self.constant_config["model_arch"] in [
            ModelArch.XGBOOST,
            ModelArch.LINEAR_OFFLINE,
        ]:
            gym = MachineGym(
                model_arch=self.constant_config["model_arch"],
                model=model,
                output_features=self.constant_config["output_features"],
                input_features=hyper_parameters["input_features"],
                future_steps=self.constant_config["future_steps"],
                train_dataset=train_dataset,
                val_dataset=dev_dataset,
                train_batch_size=self.constant_config["train_batch_size"],
                eval_batch_size=self.constant_config["eval_batch_size"],
            )
            validation_losses = gym.train(result_dir=result_dir)
        else:
            gym = DeepGym(
                model_arch=self.constant_config["model_arch"],
                model=model,
                output_features=self.constant_config["output_features"],
                input_features=hyper_parameters["input_features"],
                future_steps=self.constant_config["future_steps"],
                train_dataset=train_dataset,
                val_dataset=dev_dataset,
                train_batch_size=self.constant_config["train_batch_size"],
                eval_batch_size=self.constant_config["eval_batch_size"],
                max_tot_iter=hyper_parameters["max_iterations"],
                learning_rate=hyper_parameters["learning_rate"],
                weight_decay=self.constant_config["weight_decay"],
                lr_scheduler_linear_decay=self.constant_config.get(
                    "lr_scheduler_linear_decay", 1
                ),
                early_stopper_criteria=self.constant_config.get(
                    "early_stopper_criteria", None
                ),
            )
            validation_losses = gym.train(
                training_phase=self.constant_config["training_phase"],
                result_dir=result_dir,
                save_model_frequently=True,
                plot=True,
            )
        if use_wandb:
            wandb.finish()
        results = {
            "loss": validation_losses["mse"],
            "status": STATUS_OK,
            "applied_hyper_paras": hyper_parameters,
            "pred_dev_errors": validation_losses,
            "wandb_run_name": run_name,
        }
        self.logger.info(f"Results of hyperopt evaluation: {results}")
        return results

    def run_forever(self):
        while True:
            self.run_trials(
                result_dir=RESULT_DIR
                / "HYPEROPT2"
                / f"{self.constant_config['model_arch'].value}_{self.start_datetime}",
            )

    @staticmethod
    def init_datasets(
        hyper_para: dict, constant_config: dict
    ) -> tuple[Dataset, Dataset, Dataset]:
        dataset = partial(
            RpmSignals,
            db_root=constant_config["db_root"],
            max_num_curves=constant_config["max_num_curves"],
            scaling_period_s=constant_config["scaling_period_s"],
            white_noise_db=constant_config["white_noise_db"],
            future_points=constant_config["future_steps"],
            input_features=hyper_para["input_features"],
            output_features=constant_config["output_features"],
            use_smooth_target=constant_config["use_smooth_target"],
            cache_returned_data=False,
        )
        if "dev_min_length_s" in constant_config.keys():
            dev_min_length_s = constant_config["dev_min_length_s"]
        elif "test_signal_length_s" in constant_config.keys():
            dev_min_length_s = (
                constant_config["test_signal_length_s"]
                + (
                    constant_config["training_phase"]
                    * constant_config["output_features"]
                )
                * 0.04
                + 4
            )
        else:
            raise NotImplementedError
        train_dataset = dataset(
            phase=ModelPhases.TRAINING,
            signal_length_s=constant_config["train_signal_length_s"],
            min_length_s=constant_config["train_min_length_s"],
        )
        dev_dataset = dataset(
            phase=ModelPhases.VALIDATION,
            signal_length_s=constant_config["test_signal_length_s"],
            min_length_s=dev_min_length_s,
        )
        test_dataset = dataset(
            phase=ModelPhases.TESTING,
            signal_length_s=constant_config["test_signal_length_s"],
            min_length_s=dev_min_length_s,
        )
        return train_dataset, dev_dataset, test_dataset

    @staticmethod
    def init_model(hyper_para: dict, constant_config: dict) -> nn.Module:
        if constant_config["model_arch"] is ModelArch.TRANSFORMER_ENCODER:
            model = Transformer(
                input_features_dim=hyper_para["input_features"],
                embedding_features=hyper_para["embedding_dim"],
                num_layers=hyper_para["num_layers"],
                dropout=hyper_para["dropout"],
                n_heads=hyper_para["n_heads"],
                pre_training_size=constant_config["training_phase"],
                future_steps=constant_config["future_steps"],
                output_dim=constant_config["output_features"],
                max_signal_length_s=300,
            )
        elif constant_config["model_arch"] is ModelArch.LSTM:
            model = Many2Many(
                input_features=hyper_para["input_features"],
                num_layers=hyper_para["num_layers"],
                hidden_dim=hyper_para["hidden_dim"],
                output_dim=constant_config["output_features"],
                dropout=constant_config["dropout"],
            )
        elif constant_config["model_arch"] is ModelArch.LINEAR_OFFLINE:
            model = LinearOffline(
                input_features=hyper_para["input_features"],
                output_features=constant_config["output_features"],
                future_steps=constant_config["future_steps"],  # ToDo
                alpha=hyper_para["alpha"],
                solver="cholesky",
            )
        elif constant_config["model_arch"] is ModelArch.DLINEAR:
            model = DecompLinear(
                seq_len=hyper_para["input_features"],
                pred_len=constant_config["future_steps"],
                individual=True,
                enc_in=1,
            )
        elif constant_config["model_arch"] is ModelArch.TRANSFORMER_TSF:
            model = TransformerTSFv2(
                layer_dim_val=hyper_para["embedding_dim"],
                n_encoder_layers=hyper_para["num_layers"],
                n_decoder_layers=hyper_para["num_layers"],
                n_heads=hyper_para["n_heads"],
                dec_seq_len=hyper_para["input_features"],
                dropout=constant_config["dropout"],
                out_seq_len=1,
            )
        elif constant_config["model_arch"] is ModelArch.XGBOOST:
            model = XGBoostTSF(
                n_estimators=hyper_para["n_estimators"],
                max_depth=hyper_para["max_depth"],
                subsample_baselearner=hyper_para["subsample"],
                gamma=hyper_para["gamma"],
                min_child_weight=hyper_para["min_child_weight"],
                learning_rate=hyper_para["learning_rate"],
                reg_lambda=hyper_para["reg_lambda"],
                future_steps=constant_config["future_steps"],
            )
        elif constant_config["model_arch"] is ModelArch.CUSTOM_MODEL:
            model = YourCustomModel()
        else:
            raise ValueError(constant_config["model_arch"])
        return model

    @convert("result_dir", converter=Path)
    def run_trials(self, result_dir: PathLike):
        filepath = result_dir / "my_model.hyperopt"
        trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
        max_trials = 2  # initial max_trials. put something small to not have to wait
        try:  # try to load an already saved trials object, and increase the max
            trials = dill.load(open(filepath, "rb"))
            self.logger.info("Found saved Trials! Loading...")
            # remove trials with status fail
            index = len(trials.statuses()) - 1
            while index > -1:
                if trials.statuses()[index] == "fail":
                    trials.trials.pop(index)
                index -= 1
            max_trials = len(trials.trials) + trials_step
            self.logger.info(
                f"Rerunning from {len(trials.trials)} trials to {max_trials} (+{trials_step}) trials"
            )
        except FileNotFoundError:
            self.logger.info("No trials were found! Initializing a new object...")
            # initialize empty trials database
            trials = Trials()
        best = fmin(
            fn=self.function_to_minimize,
            space=self.search_space,
            algo=tpe.suggest,
            max_evals=max_trials,
            trials=trials,
            return_argmin=False,
        )
        self.logger.info(f"Best hyper-parameters so far: {best}")
        # save the trials object
        with open(filepath, "wb") as f:
            self.logger.info(f"ModelArch was saved at {filepath}")
            dill.dump(trials, f)

    @staticmethod
    def read_hyperopt_object(filepath: PathLike) -> hyperopt.Trials:
        with open(filepath, "rb") as f:
            trials = dill.load(f)
        best_trial = trials.best_trial
        summary = (
            "Found saved Trials! \n "
            "------------- Information --------------------\n"
            f"Number of evaluated combinations: {len(trials.trials)} \n"
            f"Best MSE loss on dev set: {best_trial['result']['loss']} \n "
            f"Corresponding hyperparas: {best_trial['result']['applied_hyper_paras']} \n"
            f"Wandb run name: {best_trial['result']['wandb_run_name']} \n"
            "------------- End Information ----------------"
        )
        logger.info(summary)
        return best_trial
