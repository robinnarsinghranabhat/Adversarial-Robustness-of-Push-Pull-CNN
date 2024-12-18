# Push-Pull layer for enhanced robustness of ConvNets [[Paper]](https://link.springer.com/article/10.1007/s00521-020-04751-8)

## First Download the Noisy-Cifar Dataset
1. Goto Project root folder (goto `Adversarial-Robustness-of-Push-Pull-CNN` folder)and run :
```bash
wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1 -O CIFAR-10-C.tar
```
2. Extract the Files inside the folder : `CIFAR-10-C`
```bash
tar -xvf CIFAR-10-C.tar
```

## 1. Without any Pushpull (Baseline Model)
### Train for 120 epochs
Command below trains a BASELINE CNN Resnet-20 architecture. Batch size is 2040, epochs is 120. 

After each iteration, output model and logs are saved at : `./experiments/resnet-cifar/resnet-20-no-pp-2048-120epoc`
Here, `resnet-20-no-pp-2048-120epoc` is name of experiment. And also folder name used to save results.

```bash
python -m train -b 2048 --arch resnet --layers 20 --name resnet-20-no-pp-2048-120epoc  --print-freq 4 --epochs 120 --use-cuda
```
### Test above model on CIFAR and Noisy-CIFAR
We use the same name used during training.   
```bash
python -m test_corruption -b 2048 --arch resnet --name resnet-20-no-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda
```
### Test above model on FGSM distorted CIFAR and FGSM distorted NOISY-CIFAR 
Specify parameter for : `fgsm-epsilon`
```bash
python -m test_corruption --fgsm-epsilon 0.01 -b 2048 --arch resnet --name resnet-20-no-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda
```
### Test above model on PGD distorted CIFAR and PGD distorted NOISY-CIFAR 
Specify parameter for : `pgd-epsilon`
```bash
python -m test_corruption_pgd --pgd-epsilon 0.01 -b 2048 --arch resnet --name resnet-20-no-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda
```



## 2. With Pushpull at just first Convolution Layer
### Train on CLEAN CIFAR
```bash
python -m train -b 2048 --arch resnet --name resnet-20-pp-2048-120epoc --pushpull --layers 20 --print-freq 4 --epochs 120 --use-cuda
```

### Test above model on FGSM distorted CIFAR and FGSM distorted NOISY-CIFAR 
Specify parameter for : `fgsm-epsilon`
```bash
python -m test_corruption  --fgsm-epsilon 0.01  -b 2048 --arch resnet --name resnet-20-pp-2048-120epoc --pushpull --layers 20 --corrupted-data-dir ./ --use-cuda
```

### Test above model on PGD distorted CIFAR and PGD distorted NOISY-CIFAR 
Specify parameter for : `pgd-epsilon`
```bash
python -m test_corruption_pgd  --pgd-epsilon 0.01  -b 2048 --arch resnet --name resnet-20-pp-2048-120epoc --pushpull --layers 20 --corrupted-data-dir ./ --use-cuda
```


## 3. With PushPull across all layers
### Train
```bash
python -m train --pushpull --pp-all -b 2048 --arch resnet --name resnet-20-all-pp-2048-120epoc --layers 20 --print-freq 4  --epochs 120 --use-cuda
```
### Test on CIFAR and NOISY-CIFAR
```bash
python -m test_corruption --pushpull --pp-all -b 2048 --arch resnet --name resnet-20-all-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda
```

### Test above model on FGSM distorted CIFAR and FGSM distorted NOISY-CIFAR 
Specify parameter for : `fgsm-epsilon`
```bash
python -m test_corruption --fgsm-epsilon 0.01 --pushpull --pp-all -b 2048 --arch resnet --name resnet-20-all-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda
```

### Test above model on PGD distorted CIFAR and PGD distorted NOISY-CIFAR 
Specify parameter for : `pgd-epsilon`
```bash
python -m test_corruption_pgd --pgd-epsilon 0.01 --pushpull --pp-all -b 2048 --arch resnet --name resnet-20-all-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda
``` 

