from __future__ import annotations

import copy
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import xgboost as xgb
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from rmp import metrics
from rmp.dataloader import ModelPhases
from rmp.early_stopping import EarlyStopping
from rmp.global_config import DEVICE, NUM_WORKERS, RESULT_DIR
from rmp.metrics import calculate_prediction_errors, calculate_relative_rmse
from rmp.models import ModelArch
from rmp.my_utils.common_types import PathLike
from rmp.my_utils.decorators import convert
from rmp.my_utils.logger import LoggerMixin, tqdm
from rmp.plotting import plot_random_signal_in_batch

logger = logging.getLogger(__name__)


class BaseGym(ABC, LoggerMixin):
    def __init__(
            self,
            model_arch: ModelArch,
            model: nn.Module,
            output_features: int,
            input_features: int,
            future_steps: int,
            train_dataset: Dataset,
            val_dataset: Dataset,
            train_batch_size: int,
            test_dataset: Dataset = None,
            eval_batch_size: int = None,
    ):
        self.model_arch = model_arch
        self.model = model
        self.output_features = output_features
        self.input_features = input_features
        self.future_steps = future_steps
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size or train_batch_size
        persistent_workers = True if NUM_WORKERS > 0 else False
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=persistent_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            persistent_workers=persistent_workers,
            num_workers=NUM_WORKERS,
        )
        if test_dataset:
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                persistent_workers=persistent_workers,
                num_workers=NUM_WORKERS,
            )

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def evaluation(self, phase: ModelPhases) -> dict | None:
        pass

    def print_stats_from_df(
            self, df: pd.DataFrame, arch: str, phase: ModelPhases, deliminator: str = "_"
    ):
        pd.set_option("display.max_columns", None)
        final_stats = {
            f"{phase.value}_final_mse": str(
                round(df[f"{arch}{deliminator}mse"].mean(), 4)
            )
                                        + "+/-"
                                        + str(round(df[f"{arch}{deliminator}mse"].std(), 4)),
            f"{phase.value}_final_rmse": str(
                round(df[f"{arch}{deliminator}rmse"].mean(), 4)
            )
                                         + "+/-"
                                         + str(round(df[f"{arch}{deliminator}rmse"].std(), 4)),
            f"{phase.value}_final_mae": str(
                round(df[f"{arch}{deliminator}mae"].mean(), 4)
            )
                                        + "+/-"
                                        + str(round(df[f"{arch}{deliminator}mae"].std(), 4)),
            f"{phase.value}_final_me": str(
                round(df[f"{arch}{deliminator}me"].mean(), 4)
            )
                                       + "+/-"
                                       + str(round(df[f"{arch}{deliminator}me"].std(), 4)),
        }
        try:
            final_stats[f"{phase.value}_final_rel_rmse"] = (
                    str(round(df[f"{arch}{deliminator}rel_rmse"].mean(), 4))
                    + "+/-"
                    + str(round(df[f"{arch}{deliminator}rel_rmse"].std(), 4))
            )

        except Exception as e:
            self.logger.warning(e)
        self.logger.info(
            f"---------FINAL STATS: {arch.upper()}-----------------\n"
            f"{final_stats}\n"
            f"---------END FINAL STATS {arch.upper()}-----------------"
        )

        df.sort_values(
            f"{arch}{deliminator}rel_rmse",
            axis=0,
            ascending=True,
            inplace=True,
            na_position="last",
        )
        self.logger.info(
            f"\n"
            f"---------TOP 5 CASES-----------------\n"
            f"{df[:5]}\n"
            f"---------WORST 5 CASES----------------\n"
            f"{df[-5:]}"
        )
        self.logger.info(f"{phase.value} finished")

    @staticmethod
    def calc_number_training_steps(
            training_phase_s: int, input_features: int, output_features: int
    ) -> int:
        if training_phase_s == 0:
            return 0
        else:
            return int(training_phase_s * 25 - input_features / output_features)

    @staticmethod
    def update_wandb(
            phase: str,
            targets: torch.Tensor,
            outputs: torch.tensor,
            return_errors: bool = False,
            seen_curves: int = 0,
            **kwargs,
    ):
        pred_errors = metrics.calculate_prediction_errors(
            y_true=targets,
            y_pred=outputs,
        )
        wandb.log(
            {
                f"{phase}_mean_mse_per_batch": pred_errors["mse"],
                f"{phase}_mean_rmse_per_batch": pred_errors["rmse"],
                f"{phase}_mean_mae_per_batch": pred_errors["mae"],
                f"{phase}_max_me_in_batch": pred_errors["me"],
            },
            step=seen_curves,
        )
        if "learning_rate" in kwargs.keys():
            wandb.log(
                {f"{phase}_learning_rate": kwargs.get("learning_rate")},
                step=seen_curves,
            )

        if return_errors:
            return pred_errors

    @staticmethod
    def update_wandb_eval(phase: str, errors: dict, seen_curves: int = 0):
        for key in errors:
            wandb.log(
                {
                    f"{phase}_mean_{key}_per_batch": errors[key],
                },
                step=seen_curves,
            )

    def _pre_eval_checking(self, phase: ModelPhases) -> DataLoader:
        if phase is ModelPhases.VALIDATION:
            loader = self.val_loader
        elif phase is ModelPhases.TESTING:
            try:
                self.__getattribute__("test_loader")
                loader = self.test_loader
            except AttributeError:
                self.logger.error(
                    f"Evaluation for {phase} cannot performed since test_loader was not initialized in BaseGym"
                )
        else:
            raise ValueError
        return loader

    def save_input_and_predictions(
            self,
            feature: np.ndarray,
            target: np.ndarray,
            output: np.ndarray,
            sample_name: str,
            result_dir: Path = RESULT_DIR
    ):

        feature = feature.squeeze()
        target = target.squeeze()
        output = output.squeeze()

        # the last value of the sliding window, i.e. the point being the prediction horizon before the target
        input_amplitude = feature[:,-1]
        self.logger.debug(
            f"{feature.shape=}, {input_amplitude.shape=}, {target.shape=}, {output.shape=}, {sample_name=} "
        )
        assert output.shape == target.shape == input_amplitude.shape
        time = 0  # ToDo add time component
        df = pd.DataFrame.from_dict(
            dict(amplitude=input_amplitude, target=target, output=output),
            dtype=np.float32,
        )
        saving_dir = result_dir / f"{self.model_arch.value}_{40 * self.future_steps}ms"
        saving_dir.mkdir(exist_ok=True)
        df.to_csv(saving_dir / f"{sample_name}.csv", index=False)
        self.logger.info(f"Input and predictions of {sample_name} where saved at {saving_dir=}")


class MachineGym(BaseGym):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation(self, phase: ModelPhases, **kwargs):
        training_phase = kwargs.get("training_phase", 0)
        self.logger.info(f"{training_phase=}")
        plot = kwargs.get("plot", False)
        result_dir = kwargs.get("result_dir", RESULT_DIR)
        loader = self._pre_eval_checking(phase=phase)
        errors_all_samples = list()
        for i, (names, features, targets) in enumerate(
                tqdm(loader, logger=self.logger, log_level=logging.INFO)
        ):
            assert (
                    features.shape[0] == targets.shape[0] == 1
            ), "batch_size greater 1 is not supported"
            sample_name = names[0]
            feature = features[0, training_phase:, :].numpy()
            target = targets[0, training_phase:, -1:].numpy().squeeze()
            output = self.model.predict_(feature)
            self.logger.debug(
                f"SHAPES: {feature.shape=}, {target.shape=}, {output.shape=}"
            )
            pred_errors = calculate_prediction_errors(y_pred=output, y_true=target)
            rel_rmse = calculate_relative_rmse(
                y_pred=output, y_true=target, future_steps=self.future_steps
            )
            merged_errors = {**pred_errors, **dict(rel_rmse=rel_rmse)}
            errors = {key: round(merged_errors[key], 4) for key in merged_errors}
            errors["curve"] = sample_name
            errors_all_samples.append(errors)

            if phase is ModelPhases.TESTING:
                self.save_input_and_predictions(
                    feature, target, output, sample_name.sample_name, result_dir=result_dir)
            if plot:
                plot_random_signal_in_batch(
                    names=names,
                    features=feature[None, :, :],
                    outputs=output[None, :, None],
                    targets=target[None, :, None],
                    future_steps=self.future_steps,
                    result_dir=result_dir,
                    seen_curves=0,
                    phase=phase,
                    max_plots=1,
                )
        df = pd.DataFrame.from_dict(errors_all_samples)
        errors = dict(
            mse=df["mse"].mean(),
            rmse=df["rmse"].mean(),
            mae=df["mae"].mean(),
            me=df["me"].mean(),
            rel_rmse=df["rel_rmse"].mean(),
        )
        if phase is ModelPhases.VALIDATION:
            self.update_wandb_eval(
                phase=phase.value,
                errors=errors,
            )
        self.print_stats_from_df(df=df, phase=phase, deliminator="", arch="")
        error_stats_filepath = result_dir / "eval_stats.csv"
        df.to_csv(error_stats_filepath)
        self.logger.info(f"Error stats were saved at {error_stats_filepath}.")
        return errors

    @convert("result_dir", converter=Path)
    def train(self, result_dir: PathLike) -> float:
        outputs_list, features, targets = [], [], []
        for i, (name, feature, target) in enumerate(
                tqdm(self.train_loader, logger=self.logger, log_level=logging.INFO)
        ):
            if feature.shape[0] != 1:
                raise ValueError
            feature_cp = copy.deepcopy(feature)
            target_cp = copy.deepcopy(target)
            self.logger.debug(f"train input size {feature.shape=}")
            features.append(feature_cp[0])
            targets.append(target_cp[0])
        features = np.concatenate(features, axis=0)
        targets = np.concatenate(targets, axis=0)
        self.logger.debug(f"SHAPES: {features.shape=}, {targets.shape=}")
        outputs = self.model.forward(features, targets)
        self.logger.debug(
            f"SHAPES: {features.shape=}, {targets.shape=}, {outputs.shape=}"
        )
        self.update_wandb(
            targets=targets, outputs=outputs, return_errors=False, phase="train"
        )
        val_mse = self.evaluation(phase=ModelPhases.VALIDATION, result_dir=result_dir)
        self.save_model(filepath=result_dir)
        return val_mse

    @convert("filepath", converter=Path)
    def save_model(self, filepath: PathLike):
        try:
            self.model.model.save_model(filepath / "model.json")
            assert self.model_arch is ModelArch.XGBOOST
        except AttributeError:
            assert self.model_arch is ModelArch.LINEAR_OFFLINE
            with open(filepath / "model.pkl", "wb") as file:
                pickle.dump(self.model.model, file)

        gym_parameters = self.__dict__.copy()
        # remove dataloaders from gym-parameters
        gym_parameters.pop("train_loader")
        gym_parameters.pop("val_loader")
        gym_parameters.pop("test_loader", None)
        gym_parameters.pop("model", None)
        torch.save(
            gym_parameters,
            filepath / "model.pth",
        )
        self.logger.info(f"{self.model_arch} was successfully saved at {filepath}.")

    @staticmethod
    @convert("filepath", converter=Path)
    def load_model(filepath: PathLike):
        loaded_dict = torch.load(filepath / "model.pth", map_location=DEVICE)
        model_arch = loaded_dict["model_arch"]
        if model_arch == ModelArch.XGBOOST.value:
            model = xgb.XGBRegressor()
            model.load_model(filepath / "model.json")
        elif model_arch == ModelArch.LINEAR_OFFLINE.value:
            with open(filepath / "model.pkl", "rb") as file:
                model = pickle.load(file)
        else:
            raise ValueError(f"{model_arch}")
        parameters = loaded_dict
        logger.info(f"{model_arch} was successfully loaded from {filepath}.")
        return model, parameters


class DeepGym(BaseGym):
    def __init__(
            self,
            *args,
            max_tot_iter: int,
            learning_rate: float = 0.001,
            weight_decay: float = 0.01,
            criterion: torch.nn.modules.loss = None,
            early_stopper_criteria: dict = None,
            lr_scheduler_linear_decay: float = 1,
            gradient_clipping: bool = False,
            **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.model = self.model.to(DEVICE)

        if early_stopper_criteria:
            self.early_train_stopper = EarlyStopping(
                early_stopper_criteria.get("patience", 1_000),
                early_stopper_criteria.get("min_delta", 0.03),
                phase=ModelPhases.TRAINING.value,
            )
            self.early_val_stopper = EarlyStopping(
                early_stopper_criteria.get("patience", 1_000),
                early_stopper_criteria.get("min_delta", 0.03),
                phase=ModelPhases.VALIDATION.value,
            )
            # default parameters meaning: after 1_000 gradient updated
            # error has to relatively decline by at least 3%
        else:
            self.early_train_stopper = None
            self.early_val_stopper = None
        self.max_tot_iter = max_tot_iter
        self.criterion = criterion or nn.MSELoss(reduction="none")
        self.gradient_clipping = gradient_clipping
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay
        )
        if not 0.8 < lr_scheduler_linear_decay <= 1:
            raise ValueError(
                f"{lr_scheduler_linear_decay=} is an invalid value: "
                f"Greater one indicates a rising lr_rate. \n"
                f"Smaller 0.8 means lr declines (too) quickly."
            )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(lr_scheduler_linear_decay ** epoch, 1e-5),
        )
        self.scaler = GradScaler()
        self.tot_iter = 0
        self.epoch = 0
        self.seen_curves = 0
        self.logger.info(
            f"Number of trainable parameters: {self.get_num_trainable_para()=}"
        )

    @staticmethod
    def unitwise_norm(x: torch.Tensor, norm_type: float = 2.0):
        if x.ndim <= 1:
            return x.norm(norm_type)
        else:
            return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)

    @staticmethod
    def adaptive_clip_grad(
            parameters: torch.Tensor,
            clip_factor: float = 0.01,
            eps: float = 1e-3,
            norm_type: float = 2.0,
    ):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        for p in parameters:
            if p.grad is None:
                continue
            p_data = p.detach()
            g_data = p.grad.detach()
            max_norm = (
                DeepGym.unitwise_norm(p_data, norm_type=norm_type)
                .clamp_(min=eps)
                .mul_(clip_factor)
            )
            grad_norm = DeepGym.unitwise_norm(g_data, norm_type=norm_type)
            clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
            new_grads = clipped_grad.where(grad_norm < max_norm, g_data, clipped_grad)
            p.grad.detach().copy_(new_grads)

    def get_num_trainable_para(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_cosine_lr_scheduler(self, **kwargs):
        self.lr_scheduler = CosineLRScheduler(
            optimizer=self.optimizer,
            t_initial=kwargs.get(
                "t_initial", 1000
            ),  # length of first maximum to minimum in epoch
            lr_min=kwargs.get("lr_min", 0.00001),
            warmup_t=kwargs.get("warmup_t", 100),  # in steps
            warmup_lr_init=kwargs.get("warmup_lr_init", 0.00001),
            cycle_limit=kwargs.get("cycle_limit", 30_000_00 // 1117),
            warmup_prefix=kwargs.get("warmup_prefix", True),
            cycle_decay=kwargs.get("cycle_decay", 0.995),
            cycle_mul=kwargs.get("cycle_mul", 1.01),
            k_decay=kwargs.get("k_decay", 1),
            initialize=True,
        )
        self.logger.info(f"lr_scheduler was set to cosine.  \n {self.lr_scheduler=}")

    @convert("filepath", converter=Path)
    def save_model(self, filepath: PathLike):
        if not (filepath.parent.is_dir() and filepath.suffix == ".pth"):
            raise ValueError(f"{filepath=}")
        gym_parameters = self.__dict__.copy()
        # remove dataloaders etc. from gym-parameters
        unwanted_keys = ["scaler", "early_val_stopper", "early_train_stopper", "criterion", "optimizer", "lr_scheduler",
                         "model", "train_loader", "val_loader", '_LoggerMixin__logger', "test_loader", "lr_scheduler"]
        for unwanted_key in unwanted_keys:
            gym_parameters.pop(unwanted_key, None)
        model_paras = self.model.input_paras
        model_paras.pop("self")
        model_paras.pop("__class__")
        # assert keys which are in model_paras and in gym are equal: otherwise major bug
        assert all([value == model_paras[key] for key, value in gym_parameters.items() if
                    key in model_paras.keys()]), f"{model_paras=} vs. {gym_parameters=}"
        final_dict = {**gym_parameters,
                      **dict(model_state_dict=self.model.state_dict()),
                      **model_paras}
        final_dict["model_arch"] = final_dict["model_arch"].value
        torch.save(final_dict, filepath)
        self.logger.info(f"{self.model_arch} was successfully saved at {filepath}.")

    @staticmethod
    @convert("filepath", converter=Path)
    def load_model(filepath: PathLike) -> tuple[dict, dict]:
        if not filepath.is_file():
            raise FileNotFoundError(f"{filepath=}")
        loaded_dict = torch.load(filepath, map_location=DEVICE)
        logger.info(f"{loaded_dict['model_arch']} was successfully loaded from {filepath}.")
        state_dict = loaded_dict.pop("model_state_dict")
        return loaded_dict, state_dict

    def evaluation(self, phase: ModelPhases, **kwargs):
        training_phase = kwargs.get("training_phase", 0)
        plot = kwargs.get("plot", False)
        on_the_fly = kwargs.get("on_the_fly", False)
        result_dir = kwargs.get("result_dir", RESULT_DIR)
        loader = self._pre_eval_checking(phase=phase)

        self.logger.info(f"{training_phase=}")
        self.model.eval()
        names, features, targets, outputs = list(), list(), list(), list()
        for i, (name, feature, target) in enumerate(tqdm(loader)):
            feature = feature.to(DEVICE)
            target = target.to(DEVICE)
            batch_size, seq_len, input_features = feature.shape
            if training_phase >= seq_len:
                raise ValueError(
                    f"{seq_len=}vs.{training_phase=},"
                    f"followed slicing would result in empty tensors"
                )
            # targets shape: (batch_size, seq_len, output_features)
            with autocast() and torch.no_grad():
                outputs.append(self.model(feature)[:, training_phase:, :].cpu())
            names.extend(list(name))
            features.append(feature[:, training_phase:, :].cpu())
            targets.append(target[:, training_phase:, :].cpu())

        if phase is ModelPhases.VALIDATION:
            outputs = torch.cat(outputs, dim=0).cpu().numpy()
            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()

            dev_pred_errors = self.update_wandb(
                seen_curves=self.tot_iter,
                phase=phase.value,
                targets=targets,
                outputs=outputs,
                return_errors=True,
            )
            if plot:
                plot_random_signal_in_batch(
                    names,
                    features,
                    targets,
                    outputs,
                    self.tot_iter,
                    result_dir=result_dir,
                    future_steps=self.future_steps,
                    phase=phase,
                )
            if on_the_fly:
                self.model.train()
            return dev_pred_errors
        elif phase is ModelPhases.TESTING:
            self.logger.info("Entered testing")
            errors_all_samples = list()
            for name, feature, target, output in zip(names, features, targets, outputs):
                feature = feature.cpu().numpy()
                target = target.cpu().numpy()
                output = output.cpu().numpy()

                pred_errors = metrics.calculate_prediction_errors(
                    y_true=target,
                    y_pred=output,
                )
                pred_errors["curve"] = name
                self.logger.debug(f"{target.shape=}, {output.shape=}")
                pred_errors["rel_rmse"] = metrics.calculate_relative_rmse(
                    y_true=target.squeeze(),
                    y_pred=output.squeeze(),
                    future_steps=self.future_steps,
                )
                self.save_input_and_predictions(
                    feature,
                    target,
                    output,
                    sample_name=name,
                    result_dir=result_dir
                )
                if plot:
                    self.logger.debug(
                        f"{feature.shape=}, {target.shape=}, {output.shape}"
                    )
                    self.logger.debug(f"{name=}")
                    plot_random_signal_in_batch(
                        [name],
                        feature,
                        target,
                        output,
                        self.tot_iter,
                        result_dir=result_dir,
                        future_steps=self.future_steps,
                        phase=phase,
                        max_plots=None,
                    )
                errors_all_samples.append(pred_errors)
            df = pd.DataFrame.from_dict(errors_all_samples)
            self.print_stats_from_df(df=df, phase=phase, deliminator="", arch="")
            error_stats_filepath = result_dir / "eval_stats.csv"
            df.to_csv(error_stats_filepath)
            self.logger.info(f"Error stats were saved at {error_stats_filepath}.")
            return None
        else:
            raise ValueError

    @convert("result_dir", converter=Path)
    def train(
            self,
            result_dir: PathLike = Path("."),
            training_phase: int = None,
            save_model_frequently=True,
            plot=True,
    ):
        if not isinstance(training_phase, int):
            raise ValueError

        self.logger.info(f"Number training sequence steps {training_phase=}")
        self.model.train()
        val_errors = {}
        while True:
            running_loss = []
            for _, (names, features, targets) in enumerate(
                    tqdm(self.train_loader, logger=self.logger, log_level=logging.INFO)
            ):
                batch_size, seq_len, _ = features.shape
                self.logger.debug(
                    f"\n loaded-shapes: \n "
                    f"{features.size()=} \n "
                    f"{targets.size()=} \n "
                )
                if training_phase >= seq_len:
                    raise ValueError(
                        f"{seq_len=}vs.{training_phase=},"
                        f"followed slicing would result in empty tensors {features.shape=}; {targets.shape=}"
                    )
                with autocast():
                    features = features.to(DEVICE)
                    targets = targets[:, training_phase:].to(DEVICE)
                    outputs = self.model(features)[:, training_phase:]
                    self.logger.debug(
                        f"\n pre-loss-shapes: \n "
                        f"{features.size()=} \n "
                        f"{targets.size()=} \n "
                        f"{outputs.size()=}"
                    )
                    assert (
                            outputs.shape == targets.shape
                    ), f"major shape bug: {outputs.shape=} vs. {targets.shape}"
                    loss = self.criterion(outputs, targets)
                    loss = loss[~torch.isnan(loss)].mean()
                self.optimizer.zero_grad(None)
                self.scaler.scale(loss).backward()
                if self.gradient_clipping:
                    # unscale the gradients of optimizer's assigned params in-place
                    self.scaler.unscale_(self.optimizer)
                    self.adaptive_clip_grad(self.model.parameters(), clip_factor=0.05)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                targets = targets.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()
                features = features.cpu().detach().numpy()
                self.seen_curves += batch_size
                self.tot_iter += 1
                mse_iter = self.update_wandb(
                    seen_curves=self.tot_iter,
                    phase=ModelPhases.TRAINING.value,
                    targets=targets,
                    outputs=outputs,
                    return_errors=True,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                )["mse"]
                running_loss.append(mse_iter)
                if self.early_train_stopper:
                    self.early_train_stopper(mse_iter)
                if self.early_val_stopper:
                    val_mse = val_errors.get("mse", 5)
                    self.logger.debug(f"{val_mse=}")
                    self.early_val_stopper(val_mse)

            self.lr_scheduler.step(epoch=self.epoch)
            if self.epoch == 0 or self.epoch % 10 == 0:
                do_plot = True if self.epoch % 4 == 0 else False
                if do_plot and plot:
                    plot_random_signal_in_batch(
                        names,
                        features[:, :, -self.output_features:],
                        targets[:, :, -self.output_features:],
                        outputs[:, :, -self.output_features:],
                        self.tot_iter,
                        result_dir=result_dir,
                        future_steps=self.future_steps,
                        phase=ModelPhases.TRAINING,
                    )
                val_errors = self.evaluation(
                    phase=ModelPhases.VALIDATION,
                    on_the_fly=True,
                    plot=plot,
                    training_phase=training_phase,
                    result_dir=result_dir,
                )
            if save_model_frequently and (self.epoch > 0 and self.epoch % 40 == 0):
                self.save_model(filepath=result_dir / f"model_{self.tot_iter}.pth")
            early_train_stop = (
                self.early_train_stopper.early_stop
                if self.early_train_stopper
                else False
            )
            early_val_stop = (
                self.early_val_stopper.early_stop if self.early_val_stopper else False
            )
            early_stop = early_train_stop or early_val_stop

            if self.tot_iter >= self.max_tot_iter or early_stop:
                # stop training loop
                dev_pred_errors = self.evaluation(
                    phase=ModelPhases.VALIDATION,
                    on_the_fly=False,
                    seen_curves=self.tot_iter,
                    plot=True,
                    training_phase=training_phase,
                    result_dir=result_dir,
                )
                self.save_model(filepath=result_dir / f"model_{self.tot_iter}.pth")
                return dev_pred_errors
            self.epoch += 1
