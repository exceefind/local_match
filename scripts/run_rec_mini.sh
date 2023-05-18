gpuid=2

cd ../

#采用LR
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s --LR_rec  --penalty_c 0.1
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s --LR_rec  --penalty_c 1.

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5 --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 3 --optim SGD --lr 0.5 --transform B_s --LR_rec  --penalty_c 2.
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 3 --optim SGD --lr 0.5 --transform B_s --LR_rec  --penalty_c 1.




#68.57
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s
#86
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 5  --n_symmetry_aug 10  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B_s

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim Adam --lr 0.001 --transform B_s
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 5  --n_symmetry_aug 10  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim Adam --lr 0.001 --transform B_s


#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 1 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --LR
#
#python eval.py --gpu 3 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt
#--distill_model  miniimagenet_good_metric_resnet12_best_44.pth

#改用与BDC一样的数据集采样 model_id = 2
# 取消掉model_t model = 3
#model 4 去掉 多余的weight_init
#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 50  --transform B --n_episodes 1000 --model_id 4 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 1 --LR

#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 2 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 1  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last --LR
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 2 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 2  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last --LR

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 2 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 3  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last --LR
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 2 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 4  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last --LR
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 2 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 5  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last --LR
#

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 2 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05 --LR --penalty_c .1  --test_times 3
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05 --LR --penalty_c 2.  --test_times 3
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05  --penalty_c 0.1 --test_times 3 --distill_model miniimagenet_confusion_resnet12_distill_2_3_gen_last.pth --LR
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05  --penalty_c 2. --test_times 3 --distill_model miniimagenet_confusion_resnet12_distill_2_3_gen_last.pth --LR
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05  --penalty_c 0.1 --test_times 5 --distill_model miniimagenet_confusion_resnet12_distill_1_3_gen_last.pth --LR
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05  --penalty_c 2. --test_times 5 --distill_model miniimagenet_confusion_resnet12_distill_1_3_gen_last.pth --LR
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 4 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05  --penalty_c 0.1 --test_times 3  --LR
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 4 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05  --penalty_c 2.  --test_times 3  --LR
#

# wd_test + lr + optim
#68.36
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim Adam --lr 0.005
#67.89
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 1 --optim Adam --lr 0.005
#67.81
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim Adam --lr 0.005
#67.87
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim Adam --lr 0.005
#68.95(bn+relu) -->  __
#69.10  best
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5
#67.98
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 1 --optim SGD --lr 0.1
#67.66
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim SGD --lr 0.1
#67.70
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim SGD --lr 0.1


## 85.94
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim Adam --lr 0.005
##85.84
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0005 --test_times 1 --optim Adam --lr 0.005
##85.57
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim Adam --lr 0.005
##85.52
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.1
##85.75
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0005 --test_times 1 --optim SGD --lr 0.1
##85.87
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim SGD --lr 0.1


#85.85
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim Adam --lr 0.001
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0005 --test_times 1 --optim Adam --lr 0.001

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim Adam --lr 0.001

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 1 --optim SGD --lr 0.5

#85.68
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 1 --optim SGD --lr 0.5

#86.09
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 3 --optim SGD --lr 0.5
#86.24
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.05 --test_times 1 --optim SGD --lr 0.5

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.1


#best
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim SGD --lr 0.5
#try:
#86.19
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5
#85.63
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim SGD --lr 0.1
#86.21
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 1.
#86.14
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 1 --optim SGD --lr 1.
#86.22
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 1.
#86.27
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 1.
#86.23
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0005 --test_times 1 --optim SGD --lr 10.

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim SGD --lr 10.

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 1. --penalty_c 1. --LR_rec

#86.02
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 1. --transform B_s
#76.53
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 1 --optim SGD --lr 0.1 --transform B_s


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 1 --transform B_s
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 5 --optim SGD --lr 0.5 --transform B_s
#86.22
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B_s
#86.16
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim SGD --lr 0.5 --transform B_s


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 1. --transform B


python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s --Loss_ablation 0
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s --Loss_ablation 1
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s --Loss_ablation 2
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5 --transform B_s --Loss_ablation 3
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B_s --Loss_ablation 0
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B_s --Loss_ablation 1
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B_s --Loss_ablation 2
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B_s --Loss_ablation 3
#





#85.95
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim SGD --lr 0.5
#85.65
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0005 --test_times 3 --optim SGD --lr 0.5

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.00005 --test_times 1 --optim SGD --lr 0.5

#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim SGD --lr 0.5




#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.05 --test_times 5


#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 1 --model_type best --prompt



#Good Embedding
#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 20  --transform B --n_episodes 1000 --model_id 11 --method confusion --batch_size 64 --reduce_dim 640 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 1 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 5 --embeding_way GE

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 20  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 10 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 170  --milestones 100 150 --k_gen 1  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --embeding_way GE
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640 --transform B --n_episodes 200 --model_id 11 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1 --n_symmetry_aug 5 --model_type last --prompt --embeding_way GE --wd_test 0.001 --optim SGD --lr 0.1
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640 --transform B --n_episodes 200 --model_id 11 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5 --n_symmetry_aug 5 --model_type last --prompt --embeding_way GE --wd_test 0.001 --optim SGD --lr 0.1



