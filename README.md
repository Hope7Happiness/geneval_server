其实直接参考https://github.com/djghosh13/geneval/issues/12
应该就行了


`cd geneval && ./evaluation/download_models.sh ../pretrained`

`conda create -n geneval python==3.8.10`

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install networkx==2.8.8 [in case you get networkx error]
pip install open-clip-torch==2.26.1
pip install clip-benchmark
pip install -U openmim
pip install einops
python -m pip install lightning
pip install diffusers transformers
pip install tomli
pip install platformdirs
pip install --upgrade setuptools
```

change to project directory;

```
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv; git checkout 1.x
MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .
cd ..
```

change to project directory;

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .
```

Fix `module 'torch' has no attribute 'xpu'`:

```
pip uninstall -y diffusers
pip install "diffusers<=0.25.0" "huggingface_hub==0.19.4" "transformers>=4.36"
```

