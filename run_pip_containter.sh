srun --gres=gpu:1 --pty singularity shell --nv ./containers/firstim/
pip install -r requirements.txt
