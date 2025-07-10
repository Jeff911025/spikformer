pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu111
pip install cupy-cuda12x
pip install timm==0.5.4 pyyaml
pip install git+https://github.com/fangwei123456/spikingjelly.git@0.0.0.0.12
conda install -c conda-forge cudatoolkit=11.1
pip install tensorboard
pip install packaging



export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


CUDA_VISIBLE_DEVICES=6 python train.py --prune --prune-ratio 0.1  --resume ./output/train/20250709-132444-spikformer-32/last.pth.tar
CUDA_VISIBLE_DEVICES=6,4 torchrun --nproc_per_node=2 train.py --resume ./output/train/20250708-214500-spikformer-32/last.pth.tar
CUDA_VISIBLE_DEVICES=6 python eval.py     --resume ./pruned_spikformer.pth.tar     --batch-size 1