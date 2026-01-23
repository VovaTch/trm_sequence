python -m scripts.train --multirun --config-name=lctm_ts model.max_thought_step=8,16 model.max_sync_steps=1,2,4,8 model.recursive_gradient_cut=True,False
