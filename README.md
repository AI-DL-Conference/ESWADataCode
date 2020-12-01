# Estimate Causal Effect of Heterogeneous Response to Marketing Campaigns for Customer Targeting  
## Desccription
dataset and source_code for our paper:<br>
--Estimate Causal Effect of Heterogeneous Response to Marketing Campaigns for Customer Targeting  

## Environment And Dependencies
PyTorch>=1.0.0<br> 
Install other dependencies: `$ pip install -r requirement.txt`  

## Dataset
Due to the size of dataset, we provide the raw data using Baidu Cloud:<br>
url: [https://pan.baidu.com/s/18yRihVhbPnhzgcuOIAlSLA](https://pan.baidu.com/s/18yRihVhbPnhzgcuOIAlSLA)<br>
password: 9viz  

## Dataset Desccription
o2o.xlsx: Treatment group and Control group -- We use the original online and the F column is our indicator:<br>
0: Control and 1: Treatment  
0102.xlsx, 0304.xlsx and 0506.xlsx are the shopping transactions.  

## Dataset Partition
We have shown the details of dataset partition in the paper.  

## Dataset Preprocessing
We have shown the details of dataset preprocessing in the paper and the embeddings are in respective subdirectories:

GRU + GNN: 
Embeddings: RNN+GNN/yes22.npy and RNN+GNN/no22.npy  
Label: ./label_yes.npy and ./label_no.npy  

GRU + T: 
Embeddings: RNN+GNN/yes7.npy and RNN+GNN/no7.npy  
Label: ./label_yes.npy and ./label_no.npy  

GRU + T + GNN: 
Embeddings: RNN+GNN/yes23.npy and RNN+GNN/no23.npy  
Label: ./label_yes.npy and ./label_no.npy

## Network Learning
Please refere to GAE:  
Paper: T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)  
Code: [GAE](https://github.com/zfjsail/gae-pytorch)(https://github.com/tkipf/gae)  

## Usage
For RNN+GNN:
```
$ cd GRU+GNN/
$ python main.py 
```

For RNN+T:
```
$ cd RNN+T/
$ python main.py 
```

For RNN+GNN+T:
```
$ cd RNN+GNN+T/
$ python main.py 
```

## Reference
<br>Any comments and feedback are appreciated.
