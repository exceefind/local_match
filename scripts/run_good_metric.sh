cd ../

#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 2 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 200 --model_id 14 --method good_metric --batch_size 64 --gama 0.1

#     running
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.1 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 200 --model_id 14 --method good_metric --batch_size 64 --gama 0.1 --drop_rate 0.5 --milestones 80 120 150

#修改提高LR的正则化 , 去掉norm 回退到至少avgpool ,

#14
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 200 --model_id 14 --method good_metric --batch_size 64 --reduce_dim 32 --gama 0.1 --drop_rate 0.1 --milestones 80 120 150

#15
#python train_pretrain.py --gpu 5 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 200 --model_id 15 --method good_metric --batch_size 128 --reduce_dim 64 --gama 0.1 --drop_rate 0.1 --milestones 80 120 150

#18  采用FPN的方式: [2,3]  server 16
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.1 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 600 --model_id 18 --method good_metric --batch_size 64 --reduce_dim 64 --drop_rate 0.5 --milestones 80 120 150 --FPN_list 2 3 --n_aug_support_samples 1 --penalty_c 1 --test --stop_grad  --t_lr 1e-3  --learnable_alpha

#19: stop_grad:
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B2 --n_episodes 100 --model_id 19 --method good_metric --batch_size 64 --reduce_dim 64 --drop_rate 0.5 --milestones 100 150 --FPN_list 2 3 --n_aug_support_samples 5 --stop_grad --penalty_c 1 --test --no_save_model

# 20  transform B
#python train_pretrain.py --gpu 6 --MultiStepLr --lr 0.1 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 600 --model_id 20 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 80 120 150 --FPN_list 2 3 --n_aug_support_samples 1 --stop_grad --penalty_c .1 --test --t_lr 1e-3

#python eval.py --gpu 2 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --model_id 20 --FPN_list 2 3  --method good_metric --reduce_dim 128   --continue_pretrain --n_shot 1 --penalty_c 1


#17:
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.1 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 600 --model_id 17 --method good_metric --batch_size 64 --reduce_dim 64  --drop_rate 0.5 --milestones 80 120 150 --FPN_list 1 3 --penalty_c 1 --test --stop_grad --t_lr 1e-3
#

#11: [1,3]  dim=128 learnable [100 150]*0.1
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --model_id 11 --method good_metric --batch_size 64 --reduce_dim 128  --drop_rate 0.5 --milestones 100 150 --FPN_list 1 3 --penalty_c 1 --test --stop_grad --t_lr 1e-3 --learnable_alpha
#python eval.py --gpu 3 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --model_id 11  --method good_metric --reduce_dim 128   --continue_pretrain --n_shot 5 --penalty_c 1  --learnable_alpha --FPN_list 1 3


# 12: [2,3] dim = 128 learnable_alpha
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 12 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 --FPN_list 2 3 --n_aug_support_samples 1 --stop_grad --penalty_c 1 --test --t_lr 1e-3 --learnable_alpha --num_workers 4
#python eval.py --gpu 6 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --model_id 12  --method good_metric --reduce_dim 128   --continue_pretrain --n_shot 1 --penalty_c 1  --learnable_alpha --FPN_list 2 3


# 13 对比12,没有FPN
#python train_pretrain.py --gpu 6 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 13 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c 1 --test --t_lr 1e-3 --learnable_alpha
#python eval.py --gpu 6 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --model_id 13  --method good_metric --reduce_dim 128   --continue_pretrain --n_shot 1 --penalty_c 1  --learnable_alpha


#14: 对比于20 采用了更多的增强  rotate
#python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B2 --n_episodes 1000 --model_id 14 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c 1 --test --t_lr 1e-3
#python eval.py --gpu 4 --val_freq 1 --test_LR --n_aug_support_samples 1 --transform B2 --n_episodes 1000 --model_id 14  --method good_metric --reduce_dim 128 --continue_pretrain --n_shot 1 --penalty_c 1


#15 dim = 256
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 15 --method good_metric --batch_size 64 --reduce_dim 256 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c 1 --test --t_lr 1e-3
#python eval.py --gpu 2  --continue_pretrain  --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 15 --method good_metric --reduce_dim 256  --n_aug_support_samples 1 --stop_grad --penalty_c 1


#21
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 21 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 --FPN_list 2 3 --n_aug_support_samples 1  --penalty_c 1 --test --t_lr 1e-3 --idea_variant
#python eval.py --gpu 0 --continue_pretrain --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 21 --method good_metric --reduce_dim 128 --FPN_list 2 3 --n_aug_support_samples 1  --penalty_c 1  --idea_variant


# 22 BDC + FPN 是否有提升
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 22 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 --FPN_list 2 3 --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea bdc
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 5 --test_LR --transform B --n_episodes 1000 --model_id 22 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea bdc --no_save_model
#python eval.py --gpu 1 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 22 --method good_metric --reduce_dim 128 --FPN_list 2 3 --n_aug_support_samples 1 --stop_grad --penalty_c .1  --learnable_alpha --num_workers 2 --idea bdc --continue_pretrain --n_shot 5


#24  idea_variant + normalize dim=1
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.1 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 24 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 1 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test
#python eval.py --gpu 5 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 24 --method good_metric  --learnable_alpha --penalty_c 1 --idea_variant --normalize_feat --FPN_list 2 3

#24 cover + 27
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 24 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 1 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 27 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 1 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test
#python eval.py --gpu 1 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 27 --method good_metric  --learnable_alpha --penalty_c 1 --idea_variant --normalize_feat --FPN_list 2 3

#28: normalize as deepbdc
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 28 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 1 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --normalize_bdc --FPN_list 2 3 --gama 0.1 --test

#29  rotate 15
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 150 --val_freq 1 --test_LR --transform B2 --n_episodes 1000 --model_id 29 --method good_metric --batch_size 64 --reduce_dim 192 --drop_rate 0.5 --milestones 80 120 --n_aug_support_samples 10 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test
#python eval.py --gpu 7 --continue_pretrain --reduce_dim 192 --transform B --n_episodes 1000 --model_id 29 --method good_metric  --learnable_alpha --penalty_c 1 --idea_variant --normalize_feat --FPN_list 2 3 --n_aug_support_samples 10


#30:
#python train_pretrain.py --gpu 5 --MultiStepLr --lr 0.05 --max_epoch 150 --val_freq 5 --test_LR --transform B --n_episodes 1000 --model_id 30 --method good_metric --batch_size 64 --reduce_dim 192 --drop_rate 0.5 --milestones 80 120 --n_aug_support_samples 10 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 4 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test
#python eval.py --gpu 7 --continue_pretrain --reduce_dim 192 --transform B --n_episodes 1000 --model_id 30 --method good_metric  --learnable_alpha --penalty_c 1 --idea_variant --normalize_feat --FPN_list 2 3 --n_aug_support_samples 10 --n_shot 5


#31 no diag
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 150 --val_freq 5 --test_LR --transform B --n_episodes 1000 --model_id 31 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 80 120 --n_aug_support_samples 10 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 4 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test --no_diag

#32  confusion :
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 32 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 1 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test --confusion

#32 cover
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 32 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 10 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test --confusion --k_c 64
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 32 --method good_metric  --learnable_alpha --penalty_c 1 --idea_variant --normalize_feat --FPN_list 2 3 --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 64


#33
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 33 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 10 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --normalize_feat --FPN_list 2 3 --gama 0.1 --test --confusion --k_c 20
#python eval.py --gpu 4 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 33 --method good_metric  --learnable_alpha --penalty_c 1 --idea_variant --normalize_feat --FPN_list 2 3 --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 20

#ID 35
#python train_pretrain.py --gpu 5 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 35 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150 180 --n_aug_support_samples 10 --stop_grad --penalty_c 1 --t_lr 1e-3 --num_workers 2 --idea_variant --normalize_feat  --gama 0.1 --test --confusion --k_c 8

# 36
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 36 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150 180 --n_aug_support_samples 10 --stop_grad --penalty_c 1 --t_lr 1e-3 --num_workers 2 --idea_variant --normalize_feat  --gama 0.1 --test --confusion --k_c 8 --learnable_alpha --FPN_list 2 3

# 38
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 38 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 8 --learnable_alpha --FPN_list 2 3

#39
#python train_pretrain.py --gpu 5 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 39 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant --normalize_feat  --gama 0.1 --test --confusion --k_c 4

#40: 在good metric中的confusion尝试引入drop，已注释
#python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 40 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 10 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant --normalize_feat  --gama 0.1 --test --confusion --k_c 8

# 41
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 41 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 10 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant --normalize_feat  --gama 0.1 --test --confusion --k_c 8

#42： 引入了confusion drop
#python train_pretrain.py --gpu 5 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 42 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 10 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant --normalize_feat  --gama 0.1 --test --confusion --k_c 4

#43 confusion drop rate = 0.3
#python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 43 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 10 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant --normalize_feat  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0.3

#44
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 44 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5 --stop_grad --penalty_c 1. --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0.2

#44 cover: k shot 验证
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 2 --test_LR --transform B --n_episodes 1000 --model_id 44 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5 --stop_grad --penalty_c 1. --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0 --n_shot 5

#ID 45
#python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 2 --test_LR --transform B --n_episodes 1000 --model_id 45 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0.1 --n_shot 1 --n_symmetry_aug 5

#ID 46
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 46 --method good_metric --batch_size 64 --reduce_dim 196 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 3 --confusion_drop_rate 0.3 --n_shot 1 --n_symmetry_aug 5

#ID 47
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 47 --method good_metric --batch_size 64 --reduce_dim 196 --drop_rate 0.5 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 3 --confusion_drop_rate 0.3 --n_shot 1 --n_symmetry_aug 1 --ths_confusion 0.99

#48
#python train_pretrain.py --gpu 3 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 50 --test_LR --transform B --n_episodes 1000 --model_id 48 --method good_metric --batch_size 64 --reduce_dim 196 --drop_rate 0.5 --milestones 100 150 180  --n_aug_support_samples 10 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion  --confusion_drop_rate 0.3 --n_shot 1 --n_symmetry_aug 1 --ths_confusion 0.9

#49 更大的ths_confusion + 去掉所有aug
#python train_pretrain.py --gpu 5 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 49 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion  --confusion_drop_rate 0 --n_shot 1 --n_symmetry_aug 1 --ths_confusion 0.95

#50  一致性的增强 即去掉colorJitter
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 1 --test_LR --transform A --n_episodes 1000 --model_id 50 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion  --n_shot 1 --n_symmetry_aug 1 --k_c 4 --confusion_drop_rate 0.3

#51
#python train_pretrain.py --gpu 5 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 51 --method good_metric --batch_size 64 --reduce_dim 96 --drop_rate 0.1 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion  --confusion_drop_rate 0 --n_shot 1 --n_symmetry_aug 1 --confusion --k_c 3

#52
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 100 --test_LR --transform B --n_episodes 1000 --model_id 52 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.1 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion  --confusion_drop_rate 0 --n_shot 1 --n_symmetry_aug 1 --confusion --k_c 3

#53
#cover: 去掉了confusion 添加了dropout
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 53 --method good_metric --batch_size 64 --reduce_dim 16 --drop_rate 0.5 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test  --confusion_drop_rate 0 --n_shot 1 --n_symmetry_aug 1

#54  transform A
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 1 --test_LR --transform A --n_episodes 1000 --model_id 54 --method good_metric --batch_size 64 --reduce_dim 16 --drop_rate 0 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test  --confusion_drop_rate 0 --n_shot 1 --n_symmetry_aug 1 --confusion --k_c 3

#55
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 55 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0 --n_shot 1 --temp_element

#56
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 56 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0 --n_shot 1 --temp_element

#57
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 57 --method good_metric --batch_size 64 --reduce_dim 36 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0 --n_shot 1

#58
#python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 58 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 2 --confusion_drop_rate 0 --n_shot 1

#59:  VAL : 66.49, TEST: 62.70
#python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 59 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0 --n_shot 1

#60  VAL:   TEST:
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 60 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --n_shot 1

#61
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 61 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --n_shot 1 --confusion --k_c 1 --confusion_drop_rate 0

#65    高度混淆16
#VAL : 66.70  LAST TEST (10 AUG): 64.21
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 65 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 10 --stop_grad --penalty_c 1. --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion --confusion_drop_rate 0 --k_c 16

#66: 修改了confusion drop的定义.
# 用更强的feat map drop 代替 整体的drop
#66 cover : drop_c 0.3--> 0.5  去掉任何aug
#VAL : 66.79  TEST: 64.21 81.81
#python train_pretrain.py --gpu 3 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 66 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion --confusion_drop_rate 0.5 --k_c 4

#67: 修改了confusion drop的定义.
#VAL(10 AUG): 69.54   TEST(10): 64.57   79.94
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 67 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion --confusion_drop_rate 0.5 --k_c 4

#68 关联的dropout2d：
# VAL：61.95
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 68 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion_drop_rate 0.25

#69
#python train_pretrain.py --gpu 3 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 68 --method good_metric --batch_size 64 --reduce_dim 196 --drop_rate 0 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion_drop_rate 0.2 --confusion --k_c 4

#添加temperature平滑output，在采用confusion threshold ，
#更小的temperature+smoothing
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 69 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion --confusion_drop_rate 0.5 --k_c 4  --ths_confusion 0.9

#70：  尝试在每个feat map上进行向量方式的归一化
#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 70 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion --confusion_drop_rate 0.5 --k_c 4  --ths_confusion 0.9

#71
python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 71 --method good_metric --batch_size 64 --reduce_dim 64 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion --confusion_drop_rate 0.1  --k_c 4

#python train_pretrain.py --gpu 3 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 70 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion --confusion_drop_rate 0.1  --ths_confusion 1 --temperature 2.0


#CUB
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 2 --test_LR --transform B --n_episodes 1000 --method good_metric --batch_size 64 --reduce_dim 196 --drop_rate 0.5 --milestones 120 170  --n_aug_support_samples 5 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 3 --confusion_drop_rate 0.3 --n_shot 1 --n_symmetry_aug 1 --dataset cub

#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 2 --test_LR --transform B --n_episodes 1000 --method good_metric --batch_size 64 --reduce_dim 196 --drop_rate 0.5 --milestones 120 170  --n_aug_support_samples 5 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 8 --confusion_drop_rate 0.3 --n_shot 1 --n_symmetry_aug 5 --dataset cub --model_id 1

#CUB resnet 18  img = 224 and ths_confusion = 0.8
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 100 --test_LR --transform B --n_episodes 1000 --method good_metric --batch_size 64 --reduce_dim 196 --drop_rate 0.5 --milestones 120 170  --n_aug_support_samples 10 --stop_grad --penalty_c .5 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 2 --confusion_drop_rate 0.3 --n_shot 1 --n_symmetry_aug 10 --dataset cub --model resnet18 --img_size 224 --ths_confusion 0.8

#Tiered ImageNet
#python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.05 --max_epoch 100 --val_freq 1 --test_LR --transform B --n_episodes 1000 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 40 70  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 16  --n_shot 1 --n_symmetry_aug 1 --dataset tieredimagenet



#25  idea_variant + flatten_fpn
#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 25 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 1 --stop_grad --penalty_c 1 --t_lr 1e-3 --num_workers 2 --idea_variant --flatten_fpn --FPN_list 2 3
#python eval.py --gpu 0 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 25 --method good_metric  --reduce_dim 128  --n_aug_support_samples 1 --stop_grad --penalty_c 1  --num_workers 2 --idea_variant  --continue_pretrain --flatten_fpn

#26 :dim = 160

#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 26 --method good_metric --batch_size 64 --reduce_dim 160 --drop_rate 0.5 --milestones 100 150 --n_aug_support_samples 1 --stop_grad --penalty_c 1 --t_lr 1e-3 --learnable_alpha --num_workers 2 --idea_variant --FPN_list 2 3 --gama 0.1 --test

# test :
#python eval.py --gpu 6 --val_freq 1 --test_LR --n_aug_support_samples 5 --transform B --n_episodes 1000 --model_id 20 --FPN_list 2 3  --method good_metric --reduce_dim 128   --continue_pretrain --n_shot 1 --penalty_c 1

#
#python eval.py --gpu 2 --val_freq 1 --test_LR --n_aug_support_samples 5 --transform B --n_episodes 1000 --model_id 17 --FPN_list 1 3  --method good_metric --reduce_dim 64   --continue_pretrain --n_shot 1 --penalty_c 1 --stop_grad

#python eval.py --gpu 1 --test_LR --val_freq 1 --transform B --n_episodes 1000 --model_id 21 --method good_metric --reduce_dim 128 --continue_pretrain --learnable_alpha --idea_variant --FPN_list 2 3 --penalty_c 1 --n_shot 5