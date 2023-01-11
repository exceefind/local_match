cd ../

#python distillation.py --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method good_metric --FPN_list 2 3 --stop_grad --model_id 20

# 65.8   83.3 optim有误.
#python distillation.py --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 160 --transform B  --method good_metric --FPN_list 2 3 --stop_grad --model_id 27 --idea_variant --normalize_feat --penalty_c 1 --n_aug_support_samples 10 --learnable_alpha

#python distillation.py --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 160 --transform B  --method good_metric --FPN_list 2 3 --stop_grad --model_id 27 --idea_variant --normalize_feat --penalty_c 1 --n_aug_support_samples 10 --learnable_alpha --max_epoch 100  --milestones 60 80 --k_gen 2

#python distillation.py --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 160 --transform B  --method good_metric --FPN_list 2 3 --stop_grad --model_id 27 --idea_variant --normalize_feat --penalty_c 1 --n_aug_support_samples 10 --learnable_alpha --max_epoch 100  --milestones 60 80 --k_gen 3

#python eval.py  --gpu 1 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method good_metric --transform B --FPN_list 2 3 --distill_model  miniimagenet_good_metric_resnet12_distill_27_1_gen.pth --n_aug_support_samples 10 --FPN_list 2 3 --stop_grad  --learnable_alpha --reduce_dim 160 --normalize_feat

#python distillation.py --gpu 2 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 192 --transform B  --method good_metric --FPN_list 2 3 --stop_grad --model_id 30 --idea_variant --normalize_feat --penalty_c 1 --n_aug_support_samples 10 --learnable_alpha --max_epoch 100  --milestones 60 80   --normalize_feat

#65.52
#python distillation.py --gpu 2 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 192 --transform B  --method good_metric --FPN_list 2 3 --stop_grad --model_id 30 --idea_variant --normalize_feat --penalty_c 1 --n_aug_support_samples 10 --learnable_alpha --max_epoch 100  --milestones 60 80 --k_gen 2  --normalize_feat

#python distillation.py --gpu 2 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 192 --transform B  --method good_metric --FPN_list 2 3 --stop_grad --model_id 30 --idea_variant --normalize_feat --penalty_c 1 --n_aug_support_samples 10 --learnable_alpha --max_epoch 100  --milestones 60 80 --k_gen 3  --normalize_feat


#python eval.py  --gpu 1 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method good_metric --transform B --FPN_list 2 3 --distill_model  miniimagenet_good_metric_resnet12_distill_30_3_gen.pth --n_aug_support_samples 10 --FPN_list 2 3 --stop_grad  --learnable_alpha --reduce_dim 192 --normalize_feat

python distillation.py --gpu 5 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 160 --transform B  --method good_metric  --stop_grad --model_id 39 --idea_variant  --penalty_c .5 --n_aug_support_samples 10 --max_epoch 170  --milestones 100 150 --k_gen 1 --normalize_feat  --confusion --k_c 4 --confusion_drop_rate 0

python distillation.py --gpu 5 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 160 --transform B  --method good_metric  --stop_grad --model_id 39 --idea_variant  --penalty_c .5 --n_aug_support_samples 10 --max_epoch 170  --milestones 100 150 --k_gen 2 --normalize_feat  --confusion --k_c 4 --confusion_drop_rate 0
