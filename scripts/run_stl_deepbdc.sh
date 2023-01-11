gpuid=2
cd ../

#python train_pretrain.py  --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --method stl_deepbdc --batch_size 64 --gama 0.1 --milestones 100 150 --t_lr 1e-3

#test stl_deepbdc
python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --penalty_c .1
python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --penalty_c 2.

#python eval_fg.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B


# self distillation
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 50  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method stl_deepbdc --k_gen 1 --penalty_c .1 --model_type last

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 50  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method stl_deepbdc --k_gen 2 --penalty_c .1 --model_type last
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 50  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method stl_deepbdc --k_gen 3 --penalty_c .1 --model_type last

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 50  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method stl_deepbdc --k_gen 4 --penalty_c .1 --model_type last

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 50  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method stl_deepbdc --k_gen 5 --penalty_c .1 --model_type last


#test distill stl_deepbdc
#python eval.py  --gpu ${gpuid} --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_distill_1_gen_last.pth --n_aug_support_samples 1
#python eval.py  --gpu ${gpuid} --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_distill_1_gen_last.pth --n_aug_support_samples 1 --penalty_c 2


