# PrognonisGraph

This project aims to create a GNN-based model for patient prognosis (See the pdf of the master thesis)

How to run the code? 

1) Install anaconda and create an environment with the file environement.yml
2) Run one of the main file by choosing the configuration in the .json file present in the configs folder
3) You can add a new dataset by following the same steps as in the data folder
3.1) load yout dataset and preprocess it 
3.2) create graph 
3.2) split your graph in train/val/test
3.3) save your graph in a pickle file
4) You can create your own networks and layers following the same structure as in the nets and layers folder
5) To search for hyperparameters, you will need to create a WandB account on the site: https://wandb.ai/ 
5.1) in the config_optimization folder you will see how to set the hyperparameters search
5.2) Set os.environ["WANDB_API_KEY"] = [your API key]
5.3) run in the temrinal :  wandb sweep config_optimisation/sweep_[name].yaml 
  
6) all the performance, log, checkpoint are save in the out folder to keep trace on your experiments
7) all the experiment tracking is also save on your wandb account


If you have any questions, please contact me on maximeb.coach@gmail.com or +32 499/72/26/52
