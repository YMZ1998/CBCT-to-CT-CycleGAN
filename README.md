# CBCT-to-CT-CycleGAN
CBCT generates pseudo CT using CycleGAN.

### [Visdom](https://github.com/facebookresearch/visdom)
To plot loss graphs and draw images in a nice web browser view

```
pip install visdom
```

```
python -m visdom.server
```

http://localhost:8097

## Environment

```bash
conda env create -f env.yml
```

```bash
conda activate lj_py
```

Export env
```bash
conda env export --no-builds --ignore-channels > env.yml
```

## Dataset
Alternatively you can build your own dataset by setting up the following directory structure:

    .
    ├── datasets                   
    |   ├── <dataset_name>         # i.e. cbct2ct
    |   |   ├── train              # Training
    |   |   |   ├── A              # Contains domain A images (i.e. cbct)
    |   |   |   └── B              # Contains domain B images (i.e. ct)
    |   |   └── test               # Testing
    |   |   |   ├── A              # Contains domain A images (i.e. cbct)
    |   |   |   └── B              # Contains domain B images (i.e. ct)


## Train, Test & Predict

```bash
python train.py
python test.py
python test_A2B.py
```

## PyInstaller Installation Guide:

### 1. Create and Activate Conda Environment

```bash
conda create --name cbct2ct python=3.9
```

```bash
conda activate cbct2ct
```

### 2. Install Required Python Packages

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyinstaller
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple SimpleITK
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
```

### 3. Use PyInstaller to Package Python Script

```bash
cd installer
pyinstaller --name CBCT2CT --onefile --icon=cbct2ct.ico CBCT2CT.py
```

### 4. Clean the Build Cache and Temporary Files

```bash
pyinstaller --clean CBCT2CT.spec
```

### 5. Run the Executable

Once the build is complete, you can run the generated `CBCT2CT.exe` with the required parameters:

```bash
CBCT2CT.exe --cbct_path ./test_data/brain.nii.gz --result_path ./result --anatomy brain
```

```bash
CBCT2CT.exe --cbct_path ./test_data/pelvis.nii.gz --result_path ./result --anatomy pelvis
```

- `--cbct_path`: Path to the input CBCT image file.
- `--result_path`: Path where the results will be saved.
- `--anatomy`: Choose a model based on anatomical region.

### 6. Deactivate and Remove Conda Environment

```bash
conda deactivate
conda remove --name cbct2ct --all
```

# Reference
[CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

[PyTorch-CycleGAN](https://github.com/YMZ1998/PyTorch-CycleGAN)