# Provable Repair of Deep Neural Network Defects by Preimage Synthesis and Property Refinement

This repository contains the codes and scripts for the paper "Provable Repair of Deep Neural Network Defects by Preimage Synthesis and Property Refinement". It also includes the full version of the paper, which provides additional experimental results and detailed experimental setups.

## Requirements

To run the code, please ensure the following dependencies are installed:

- Python 3.9.19
- PyTorch 2.3.1
- auto_LiRPA: You can install auto_LiRPA from [here](https://github.com/Verified-Intelligence/auto_LiRPA).
- This project also includes an implementation of the **α,β-CROWN complete verifier** in the folder `complete_verifier`.  
  For more details about this verifier, please refer to the [α,β-CROWN repository](https://github.com/Verified-Intelligence/alpha-beta-CROWN).

## Models
Due to capacity limitations, we uploaded all models involved in the experiments of this paper to this [anonymous link](https://figshare.com/s/fee10a0967ed91fdd6e8).


### Structure
```
.
├── model/
│   ├── acasxu/          models for experiment of global safety property violation repair (.pth)
│   ├── acasxu_nnet/     models for experiment of global safety property violation repair (.nnet)
│   ├── backdoor/        models for experiment of backdoor repair (SVHN, GTSRB, CIFAR-10 and CIFAR-100)
│   ├── mnist/           models for experiment of natural corruption repair
│   └── robustness/      models for experiment of adversarial attack repair (MNIST, GTSRB, and CIFAR-10)
```

## Reproducing the Experiments

### Reproduct repairing Natural Corruption

You can run the following command, the resulting logs will be saved in: `result/corruption/Ours/`
```
python repair_corr.py --model 9x100 --N 100 --corruption fog --exp
```

* ```<model>```: specifies the model to be repaired (e.g., 6x100, 9x100, 9x200).

* ```<N>```: specifies the number of buggy data (desired properties).

* ```<corruption>```: specifies the type of corruption.

* ```<exp>```: specifies whether recode the experimental results to the log file (**False means record**)


### Reproduct repairing Backdoor

You can run the following command, the resulting logs will be saved in: `result/backdoor/Ours`
```
python repair_back.py --dataset CIFAR10 --model VGG11 --N 100 --attack BadnetsA2O --exp
```

* ```<dataset>```: specifies the dataset (e.g., SVHN, GTSRB, CIFAR10, CIFAR100).

* ```<model>```: specifies the model to be repaired (e.g., VGG11, ResNet18).

* ```<N>```: specifies the number of buggy data (desired properties).

* ```<attack>```: specifies the type of backdoor attack (e.g., BadnetsA2O, Blend, TrojanNN, etc.).

* ```<exp>```: specifies whether recode the experimental results to the log file (False means record)


### Reproduct repairing local robustness (adversarial perturbation)

You can run the following command, the resulting logs will be saved in: `result/robustness`
```
python repair_rob.py --dataset CIFAR10 --model cnn4 --N 1 --eps 2.0 --exp False --ndims 16 --seed 0 --refine_method
```

* ```<dataset>```: specifies the dataset (e.g., MNIST, GTSRB, CIFAR10).

* ```<model>```: specifies the model to be repaired (e.g., 3x00, cnn3, cnn4).

* ```<N>```: specifies the number of local space need repair (desired properties).

* ```<eps>```: specifies the raduis of perturbation, for example, 0.1 for MNIST (with 0-1 normalization) 2.0 for GTSRB and CIFAR10 (0-255).

* ```<exp>```: specifies whether recode the experimental results to the log file (False means record)

* ```<ndims>```: specifies the dimensionality of the local space

* ```<refine_method>```: specifies the refine metric (default is 'refine_score'). Set it to 'mag' means refine the dimension based on the magnitude.

* ```<seed>```: specifies the random seed.


### Reproduct repairing global safety property

You can run the following command, the resulting logs will be saved in: `result/global_safety`
```
python repair_global_safety_p.py --net n29 --seed 0
```

* ```<net>```: specifies the model to be repaired.

* ```<refine_method>```: specifies the refine metric (default is 'refine_score'). Set it to 'mag' means refine the dimension based on the magnitude.

* ```<seed>```: specifies the random seed.



