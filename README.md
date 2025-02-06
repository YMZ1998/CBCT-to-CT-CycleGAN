# CBCT-to-CT-CycleGAN
CBCT generates pseudo CT using CycleGAN.

## visdom
```
pip install visdom
```

```
python -m visdom.server
```

http://localhost:8097

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