import os
import shutil

src = '../checkpoint'
dst = './checkpoint'
os.makedirs(dst, exist_ok=True)

for anatomy in ['pelvis', 'brain']:
    shutil.copy(os.path.join(src, anatomy, 'cbct2ct.onnx'), os.path.join(dst, anatomy+'.onnx'))
