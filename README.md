# TSECA-Conn  
TSECA-Conn is a project implementing BERT combined with GNN with the option to use connection words and a dual-stream architecture. The project contains several folders and files, including the model files, data folder, and more.  
  
## Directory Structure  
  
<pre>  
TSECA-Conn/  
│  
├── Roberta_TS_zh/          # Dual-stream model folder  
│   └── config.py           # Configuration file for hyperparameters  
├── Roberta_zh/             # Single-stream model folder  
│   └── config.py           # Configuration file for hyperparameters  
├── data/                   # Data folder  
├── preprocess/             # Preprocessing scripts  
├── result/                 # Results folder  
├── statistics/             # Statistics folder  
└── README.md               # This README file  
</pre>  

## Dependencies  
  
Before you can run this project, make sure to install the necessary dependencies:  
  
```bash  
Python==3.8  
PyTorch==1.13  
Transformers==4.27.4
```

## Quick Start
 
Follow these steps to get started with TSECA-Conn:
1. Clone or download this repository.<br/>
```git clone https://github.com/JunfengRan/TSECPE.git```
 
2. Choose whether or not to add connection words and whether to use a dual-stream model. In this example, we use connection words and the dual-stream model. You need to run the following command to execute the model:<br/>
```python main_conn.py  ```
 
3. After the model has finished running, evaluate its performance by running:<br/>
```python main_evaluate_conn.py  ```
 
## Paper
If you use our codes or your research is related to our paper, please kindly cite our paper:
XXX
