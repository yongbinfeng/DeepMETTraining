## Some code used in DeepMET trainings
(Code Taken from Jan Steggemann and Markus Seidel, modifed for the L1 MET studies)

First install the necessary packages with [MiniConda](https://docs.conda.io/en/latest/miniconda.html)
```
conda create -n METTraining python=3.7
conda install -n METTraining numpy h5py
conda install -n METTraining progressbar2
conda install -n METTraining matplotlib pandas scikit-learn
conda install -n METTraining tensorflow-gpu=1.13.1 keras=2.2.4
```
and activate the environment
```
conda activate METTraining
```
Intall a few more modules:
```
pip install mplhep
pip install tables
pip install uproot
```

Clone the code to the local repo, and check out the l1met branch
```
git clone git@github.com:yongbinfeng/DeepMETTraining.git
git checkout l1met
```
Update the `inputfile` to the location where the inputfile is saved. One inputfile can be downloaded from [here](https://cernbox.cern.ch/index.php/s/1d6aOVIO1ltxnCl)

Run the training
```
python train_ptmiss.py
```
