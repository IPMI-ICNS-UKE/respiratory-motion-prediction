import logging
from pathlib import Path

from rmp.dataloader import ModelPhases
from rmp.global_config import DATALAKE, SAVED_MODELS_DIR
from rmp.gym import BaseGym, DeepGym, MachineGym
from rmp.hyperoptimizer import Hyperoptimizer
from rmp.models import (
    DecompLinear,
    LinearOffline,
    Many2Many,
    ModelArch,
    Transformer,
    TransformerTSFv2,
    XGBoostTSF,
)
from rmp.utils.common_types import PathLike
from rmp.utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


class Eval(LoggerMixin):
    def __init__(self, future_steps: int, model_filepath: PathLike, config: dict):
        self._future_steps = future_steps
        self._model_filepath = model_filepath
        baseconfig = dict(
            train_batch_size=128,
            eval_batch_size=1,
            output_features=1,
            db_root=DATALAKE,
            max_num_curves=None,
            scaling_period_s=(0, 20),
            white_noise_db=27,
            train_signal_length_s=40,
            train_min_length_s=59,
            test_signal_length_s=None,
            use_smooth_target=True,
            cache_returned_data=False,
            training_phase_s=20,
            dev_min_length_s=30,
        )
        self._config = {**baseconfig, **config, **dict(future_steps=future_steps)}
        if baseconfig["max_num_curves"] is not None:
            self.logger.warning(
                f"Only {baseconfig['max_num_curves']} curve(s) were loaded!"
            )

    @property
    def config(self):
        return self._config

    @property
    def model_filepath(self):
        return self._model_filepath

    @property
    def future_steps(self):
        return self._future_steps

    @classmethod
    def init_lstm_480(cls):
        return cls(
            future_steps=12,
            model_filepath=SAVED_MODELS_DIR
            / "LSTM"
            / Path("noble-lion-95_480_ms_output_1/model_10386.pth"),
            config=dict(
                model_arch=ModelArch.LSTM,
                input_features=32,
            ),
        )

    @classmethod
    def init_lstm_680(cls):
        return cls(
            future_steps=17,
            model_filepath=SAVED_MODELS_DIR
            / "LSTM"
            / Path("flowing-darkness-91_680_ms_output_1/model_9234.pth"),
            config=dict(
                model_arch=ModelArch.LSTM,
                input_features=8,
            ),
        )

    @classmethod
    def init_lstm_920(cls):
        return cls(
            future_steps=23,
            model_filepath=SAVED_MODELS_DIR
            / "LSTM"
            / Path("crimson-wind-59_920_ms_output_1/model_6336.pth"),
            config=dict(
                model_arch=ModelArch.LSTM,
                input_features=4,
            ),
        )

    @classmethod
    def init_dlinear_480(cls):
        return cls(
            future_steps=12,
            model_filepath=SAVED_MODELS_DIR
            / "DLINEAR"
            / Path("worthy-tree-77_480_ms_output_1/model_2445.pth"),
            config=dict(
                model_arch=ModelArch.DLINEAR,
                input_features=450,
            ),
        )

    @classmethod
    def init_dlinear_680(cls):
        return cls(
            future_steps=17,
            model_filepath=SAVED_MODELS_DIR
            / "DLINEAR"
            / Path("gallant-vortex-94_680_ms_output_1/model_2670.pth"),
            config=dict(
                model_arch=ModelArch.DLINEAR,
                input_features=350,
            ),
        )

    @classmethod
    def init_dlinear_920(cls):
        return cls(
            future_steps=23,
            model_filepath=SAVED_MODELS_DIR
            / "DLINEAR"
            / Path("trim-salad-86_920_ms_output_1/model_2515.pth"),
            config=dict(
                model_arch=ModelArch.DLINEAR,
                input_features=225,
            ),
        )

    @classmethod
    def init_linear_offline_480(cls):
        return cls(
            future_steps=12,
            model_filepath=SAVED_MODELS_DIR
            / Path("LINEAR_OFFLINE/smooth-snowball-548_480_ms_output_1/model.pth"),
            config=dict(model_arch=ModelArch.LINEAR_OFFLINE, input_features=445),
        )

    @classmethod
    def init_linear_offline_680(cls):
        return cls(
            future_steps=17,
            model_filepath=SAVED_MODELS_DIR
            / Path("LINEAR_OFFLINE/exalted-bush-397_680_ms_output_1/model.pth"),
            config=dict(model_arch=ModelArch.LINEAR_OFFLINE, input_features=495),
        )

    @classmethod
    def init_linear_offline_920(cls):
        return cls(
            future_steps=23,
            model_filepath=SAVED_MODELS_DIR
            / Path("LINEAR_OFFLINE/silvery-donkey-312_920_ms_output_1/model.pth"),
            config=dict(model_arch=ModelArch.LINEAR_OFFLINE, input_features=495),
        )

    @classmethod
    def init_xgboost_480(cls):
        return cls(
            future_steps=12,
            model_filepath=SAVED_MODELS_DIR
            / "XGBOOST"
            / Path("desert-feather-600_480_ms_output_1/model.pth"),
            config=dict(
                model_arch=ModelArch.XGBOOST,
                input_features=175,
            ),
        )

    @classmethod
    def init_xgboost_680(cls):
        return cls(
            future_steps=17,
            model_filepath=SAVED_MODELS_DIR
            / "XGBOOST"
            / Path("smooth-glitter-835_680_ms_output_1/model.pth"),
            config=dict(
                model_arch=ModelArch.XGBOOST,
                input_features=450,
            ),
        )

    @classmethod
    def init_xgboost_920(cls):
        return cls(
            future_steps=23,
            model_filepath=SAVED_MODELS_DIR
            / "XGBOOST"
            / Path("mild-bird-785_920_ms_output_1/model.pth"),
            config=dict(
                model_arch=ModelArch.XGBOOST,
                input_features=450,
            ),
        )

    @classmethod
    def init_transformer_480(cls):
        return cls(
            future_steps=12,
            model_filepath=SAVED_MODELS_DIR
            / "TRANSFORMER_ENCODER"
            / Path("proud-brook-25_480_ms_output_1/model_14166.pth"),
            config=dict(
                model_arch=ModelArch.TRANSFORMER_ENCODER,
                input_features=8,
            ),
        )

    @classmethod
    def init_transformer_680(cls):
        return cls(
            future_steps=17,
            model_filepath=SAVED_MODELS_DIR
            / "TRANSFORMER_ENCODER"
            / Path("clean-morning-14_680_ms_output_1/model_15858.pth"),
            config=dict(
                model_arch=ModelArch.TRANSFORMER_ENCODER,
                input_features=32,
            ),
        )

    @classmethod
    def init_transformer_680_xai(cls):
        return cls(
            future_steps=17,
            model_filepath=SAVED_MODELS_DIR
            / "TRANSFORMER_ENCODER"
            / Path("noble-sea-92_680_ms_output_1/model_20016.pth"),
            config=dict(
                model_arch=ModelArch.TRANSFORMER_ENCODER,
                input_features=1,
            ),
        )

    @classmethod
    def init_transformer_920(cls):
        return cls(
            future_steps=23,
            model_filepath=SAVED_MODELS_DIR
            / "TRANSFORMER_ENCODER"
            / Path("atomic-plasma-16_920_ms_output_1/model_14364.pth"),
            config=dict(
                model_arch=ModelArch.TRANSFORMER_ENCODER,
                input_features=16,
            ),
        )

    @classmethod
    def init_transformer_v2_480(cls):
        """only eval on server feasible due to high gpu usage."""
        return cls(
            future_steps=12,
            model_filepath=SAVED_MODELS_DIR
            / Path("TRANSFORMER_K/dainty-wind-35_480_ms_output_1/model_8190.pth"),
            config=dict(
                model_arch=ModelArch.TRANSFORMER_TSF,
                input_features=150,
            ),
        )

    @classmethod
    def init_transformer_v2_680(cls):
        """only eval on server feasible due to high gpu usage."""
        return cls(
            future_steps=17,
            model_filepath=SAVED_MODELS_DIR
            / Path("TRANSFORMER_K/major-frost-190_680_ms_output_1/model_9100.pth"),
            config=dict(
                model_arch=ModelArch.TRANSFORMER_TSF,
                input_features=95,
            ),
        )

    @classmethod
    def init_transformer_v2_920(cls):
        """only eval on server feasible due to high gpu usage."""
        return cls(
            future_steps=23,
            model_filepath=SAVED_MODELS_DIR
            / Path("TRANSFORMER_K/desert-pine-258_920_ms_output_1/model_8960.pth"),
            config=dict(
                model_arch=ModelArch.TRANSFORMER_TSF,
                input_features=100,
            ),
        )

    def eval_saved_model_using_test_set(
        self,
    ):
        test_dataset = Hyperoptimizer.init_datasets(
            self.config, self.config, phases=ModelPhases.TESTING
        )

        if self.config["model_arch"] in [ModelArch.XGBOOST, ModelArch.LINEAR_OFFLINE]:
            logger.info(f"{self.model_filepath=}")
            model, loaded_model_parameters = MachineGym.load_model(
                self.model_filepath.parent
            )
            if self.config["model_arch"] is ModelArch.LINEAR_OFFLINE:
                model_cls = LinearOffline(
                    future_steps=loaded_model_parameters["future_steps"],
                    input_features=loaded_model_parameters["input_features"],
                    output_features=loaded_model_parameters["output_features"],
                )
                model_cls.model = model
                model_cls.trained = True
            elif self.config["model_arch"] is ModelArch.XGBOOST:
                model_cls = XGBoostTSF(
                    n_estimators=1,
                    max_depth=1,
                    subsample_baselearner=1,
                    gamma=1,
                    min_child_weight=1,
                    learning_rate=0.3,
                    reg_lambda=0.3,
                    future_steps=loaded_model_parameters["future_steps"],
                )
                model_cls.model = model
                model_cls.trained = True
            else:
                raise ValueError(f"{self.config['model_arch']=}")
            gym = MachineGym(
                model_arch=self.config["model_arch"],
                model=model_cls,
                output_features=loaded_model_parameters["output_features"],
                input_features=loaded_model_parameters["input_features"],
                future_steps=loaded_model_parameters["future_steps"],
                train_dataset=None,
                val_dataset=None,
                test_dataset=test_dataset,
                train_batch_size=self.config.get("train_batch_size", 1),
                eval_batch_size=self.config["eval_batch_size"],
            )
        else:
            loaded_model_parameters, state_dict = DeepGym.load_model(
                filepath=self.model_filepath
            )
            if self.config["model_arch"] is ModelArch.DLINEAR:
                model = DecompLinear(
                    seq_len=loaded_model_parameters["seq_len"],
                    pred_len=loaded_model_parameters["pred_len"],
                    individual=loaded_model_parameters["individual"],
                    enc_in=loaded_model_parameters["enc_in"],
                )
            elif self.config["model_arch"] is ModelArch.TRANSFORMER_ENCODER:
                model = Transformer(
                    input_features_dim=loaded_model_parameters["input_features_dim"],
                    embedding_features=loaded_model_parameters["embedding_features"],
                    num_layers=loaded_model_parameters["num_layers"],
                    n_heads=loaded_model_parameters["n_heads"],
                    future_steps=loaded_model_parameters["future_steps"],
                    pre_training_size=loaded_model_parameters["pre_training_size"],
                )
            elif self.config["model_arch"] is ModelArch.LSTM:
                model = Many2Many(
                    input_features=loaded_model_parameters["input_features"],
                    num_layers=loaded_model_parameters["num_layers"],
                    hidden_dim=loaded_model_parameters["hidden_dim"],
                    output_dim=1,
                    dropout=0,
                )
            elif self.config["model_arch"] is ModelArch.TRANSFORMER_TSF:
                model = TransformerTSFv2(
                    layer_dim_val=loaded_model_parameters["layer_dim_val"],
                    dec_seq_len=loaded_model_parameters["dec_seq_len"],
                    out_seq_len=loaded_model_parameters["out_seq_len"],
                    n_decoder_layers=loaded_model_parameters["n_decoder_layers"],
                    n_encoder_layers=loaded_model_parameters["n_encoder_layers"],
                    n_heads=loaded_model_parameters["n_heads"],
                    dropout=0,
                )
            else:
                raise ValueError(f"{self.config['model_arch']=} not supported.")
            model.load_state_dict(state_dict=state_dict)
            gym = DeepGym(
                model_arch=self.config["model_arch"],
                model=model,
                output_features=loaded_model_parameters["output_features"],
                input_features=loaded_model_parameters["input_features"],
                future_steps=loaded_model_parameters["future_steps"],
                train_dataset=None,
                val_dataset=None,
                test_dataset=test_dataset,
                train_batch_size=self.config.get("train_batch_size", 1),
                eval_batch_size=self.config["eval_batch_size"],
                max_tot_iter=0,
            )
        assert (
            loaded_model_parameters["input_features"] == self.config["input_features"]
        )
        assert loaded_model_parameters["future_steps"] == self.config["future_steps"]
        assert (
            loaded_model_parameters["output_features"] == self.config["output_features"]
        )
        assert loaded_model_parameters["model_arch"] == self.config["model_arch"].value
        num_training_steps = BaseGym.calc_number_training_steps(
            training_phase_s=self.config["training_phase_s"],
            input_features=loaded_model_parameters["input_features"],
            output_features=loaded_model_parameters["output_features"],
        )
        _ = gym.evaluation(
            phase=ModelPhases.TESTING,
            training_phase=num_training_steps,
            plot=True,
            result_dir=self.model_filepath.parent,
        )
        logger.info("Evaluation completed!")
