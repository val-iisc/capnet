# Sample arguments to run training code. Set argument values as necessary.
# Check the list of arguments in train.py for more options

python train.py \
	--exp exp_dir \
	--gpu 0 \
	--category chair \
	--N_VIEWS 4 \
	--print_n 100 \
	--save_n 1000 \
	--save_model_n 5000 \
	--max_epoch 10
