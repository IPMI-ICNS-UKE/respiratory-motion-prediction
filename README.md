# Respiratory motion prediction
Respiratory motion prediction (more precisely: respiratory signal prediction) means to find the respiratory amplitude value that is the prediction horizon ahead (uni-variate time-series forecasting, single point prediction). This repository provides the source code and models corresponding to our manuscript *Benchmarking real-time respiratory signal predictors in 4D SBRT* (currently under review).

The framework provides training, validation and optimization of six models for respiratory signal prediction:
- LINEAR_OFFLINE
- DLINEAR
- LSTM
- TRANSFORMER_ENCODER
- TRANSFORMER_TSF
- XGBOOST

Final evaluation should be performed using the test set. 

In our original study, we investigated three prediction horizons (480ms, 680ms and 920ms). For each horizon and each model, an optimization was performed (yielding in 18 models in total). To explore the results reported in our manuscript, an interactive dashboard is available at https://research.ipmi.uni-hamburg.de. The in-house respiratory signal database used can be downlaoded from [here](https://github.com/IPMI-ICNS-UKE/respiratory-signal-database/tree/main).

Author: Lukas Wimmert ([l.wimmert@uke.de](mailto:author_email))

## Installation
Download the database and install the python database package (_resp_db_) by following the instructions [here](https://github.com/IPMI-ICNS-UKE/respiratory-signal-database/tree/main).
Then, clone this repository to your local machine:
```bash
git clone https://github.com/IPMI-ICNS-UKE/respiratory-motion-prediction.git
```
cd into the repo and install necessary dependencies and the package itself:
```bash
pip install -r requirements.txt
pip install -e .
```
Lastly, go to _rmp/global_config.py_ and change 
```python
RESULT_DIR = Path(".../results") # dir where hyperopt results are stored
DATALAKE = Path(".../open_access_rpm_signals_master.db")  # change to path of downloaded database
```
Tested with Python 3.9.11 and Pytorch 1.11.0.



## Usage

###  Perform training, validation and optimization
- Go to _scripts/run_hyperopt.py_.
- Select one of the six implemented models and choose a prediction horizon.
- Start _scripts/run_hyperopt.py_.
- Track training and validation losses for different hyperparameter combinations with _weights&biases_ (wandb). 
- Based on that, choose the best-performing model.

We recommend creating a free wandb account [(see here)](https://docs.wandb.ai/quickstart) and using it for loss tracking.
### Evaluation using trained models
To reproduce our achieved results or to evaluate a newly trained model on the test set:
- Go to _scripts/run_eval_model.py_.
- Select a model and a prediction horizon by selecting a class method of Eval
  - Those methods follow the structure 'init_ModelArch_PredictionHorizon' (PredictionHorizon in ms)  
- Start _scripts/run_eval_model.py_.
- Evaluation might take some time and depending on the model requires much GPU capacity.

### Include your own model to the pipeline
- Go to _rmp/models.py_.
- Add your model logic to the class YourCustomModel and define its forward function.
- Go to _scripts/run_hyperopt.py_.
- Select ModelArch.CUSTOM_MODEL and choose a prediction horizon.
- Start _scripts/run_hyperopt.py_.


## License
[MIT](https://choosealicense.com/licenses/mit/)
