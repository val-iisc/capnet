#!/bin/sh

# Sample arguments to run metrics code. Make sure that the corresponding
# experiment directory and snapshot folders exist. Use --visualize in place of
# --tqdm to visualize outputs. Check the arguments for more options.

python metrics.py \
	--exp exp_dir \
	--setup projection \
	--gpu 1 \
	--category chair \
	--eval_set val \
	--snapshot latest \
	--visualize
