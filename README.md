# Push-Pull layer for enhanced robustness of ConvNets [[Paper]](https://link.springer.com/article/10.1007/s00521-020-04751-8)

## Creating Python environment if not present (Optional)
```bash
## commands below creates a python virtual environment called  "myenv" and installs all libraries inside it.
python3 -m venv myenv
source ./myenv/bin/activate
pip install --upgrade pip
pip install jupyter ipython ipykernel magic-wormhole
ipython kernel install --user --name=myenv

pip3 install torch torchvision torchaudio
pip3 install tensorboard_logger tensorflow
```

## Project Setup
```bash
# Clone the Repo
git clone https://github.com/robinnarsinghranabhat/Adversarial-Robustness-of-Push-Pull-CNN.git

# Goto to Project Root 
cd ./Adversarial-Robustness-of-Push-Pull-CNN

# Download the Corrupted-Cifar Dataset in above folder
wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1 -O CIFAR-10-C.tar

# Extract the Files inside the folder : `CIFAR-10-C`
tar -xvf CIFAR-10-C.tar
```

## 1. Without any Pushpull (Baseline Model)
### Train for 120 epochs
Command below trains a BASELINE CNN Resnet-20 architecture. In example below, Batch size is 2040, epochs is 120. 

After each iteration, output model and logs are saved at : `./experiments/resnet-cifar/resnet-20-no-pp-2048-120epoc`
Here, `resnet-20-no-pp-2048-120epoc` is name of experiment. And also folder name used to save results.

```bash
python -m train_direct -b 2048 --arch resnet --layers 20 --name resnet-20-no-pp-2048-120epoc  --print-freq 4 --epochs 120 --use-cuda
```
### Test above model on CIFAR and Noisy-CIFAR
We use the same name used during training.   
```bash
python -m test_corruption -b 2048 --arch resnet --name resnet-20-no-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda --geom-transform
```
### Test above model on FGSM distorted CIFAR and FGSM distorted NOISY-CIFAR 
Specify parameter for : `fgsm-epsilon`
```bash
python -m test_corruption --fgsm-epsilon 0.01 -b 2048 --arch resnet --name resnet-20-no-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda
```
### Test above model on PGD distorted CIFAR and PGD distorted NOISY-CIFAR 
Specify parameter for : `pgd-epsilon`
```bash
python -m test_corruption_pgd --pgd-epsilon 0.01 -b 2048 --arch resnet --name resnet-20-no-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda --geom-transfom
```



## 2. With Pushpull at just first Convolution Layer
### Train on CLEAN CIFAR
```bash
python -m train_direct -b 2048 --arch resnet --name resnet-20-pp-2048-120epoc --pushpull --layers 20 --print-freq 4 --epochs 120 --use-cuda
```
### Test above model on CIFAR and Noisy-CIFAR
We use the same name used during training.   
```bash
python -m test_corruption -b 2048 --arch resnet --name resnet-20-pp-2048-120epoc --pushpull --layers 20 --corrupted-data-dir ./ --use-cuda
```
### Test above model on FGSM distorted CIFAR and FGSM distorted NOISY-CIFAR 
Specify parameter for : `fgsm-epsilon`
```bash
python -m test_corruption  --fgsm-epsilon 0.01  -b 2048 --arch resnet --name resnet-20-pp-2048-120epoc --pushpull --layers 20 --corrupted-data-dir ./ --use-cuda
```

### Test above model on PGD distorted CIFAR and PGD distorted NOISY-CIFAR 
Specify parameter for : `pgd-epsilon`
```bash
python -m test_corruption_pgd  --pgd-epsilon 0.01  -b 2048 --arch resnet --name resnet-20-pp-2048-120epoc --pushpull --layers 20 --corrupted-data-dir ./ --use-cuda --geom-transform
```


## 3. With PushPull across all layers
### Train
```bash
python -m train_direct --pushpull --pp-all -b 2048 --arch resnet --name resnet-20-all-pp-2048-120epoc --layers 20 --print-freq 4  --epochs 120 --use-cuda
```
### Test on CIFAR and NOISY-CIFAR
```bash
python -m test_corruption --pushpull --pp-all -b 2048 --arch resnet --name resnet-20-all-pp-2048-120epoc --layers 20 --corrupted-data-dir ./ --use-cuda --geom-transform
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

## 4. Wider RESNET expansion 4, No push-pull
- Training
``bash
python -m train_direct -b 2048 --arch resnet --layers 20 --name resnet-20-no-pp-2048-120epoc-exp-4  --print-freq 4 --epochs 120 --u
se-cuda --expansion 4
```

- Testing
```bash
python -m test_corruption -b 2048 --arch resnet --name resnet-20-no-pp-2048-120epoc-exp-4 --layers 20 --corrupted-data-dir ./ --use-cuda 
```


## 5. Wider RESTNET expansion 4, All Push-Pull
```bash
python -m train_direct --pushpull --pp-all -b 2048 --arch resnet --name resnet-20-all-pp-2048-120epoc-exp-4 --layers 20 --print-freq 4  --epochs 120 --use-cuda --expansion 4
```