# kaggle_RRVF
Mini-project for the Kaggle competition Recruiting Restaurant Visitor Forecasting:  
https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/

## Install environment
```bash
conda env create -f environment.yml
```

## Fit the model
There are different files `fit_*.py` which build different models.  
For instance, to create a model for each latency value:  
```bash
python fit_fixed_latency.py
```
All the "fit scripts" create a `.pkl` file with the model and the `.csv` to be submitted to Kaggle.  
The scripts assume that there is a `data` directory with 2 files in it:  
- `train.tsv`  
- `test.tsv`
 

## Configurations
Configurables are defined in `defines.py`