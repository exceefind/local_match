gpuid=7

cd ../../

#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 100 --val_freq 1 --test_LR --transform B --n_episodes 1000 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 40 70  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant   --test --confusion --k_c 16  --n_shot 1 --n_symmetry_aug 1 --dataset tieredimagenet

python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 100 --val_freq 10 --test_LR --transform B --n_episodes 1000 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 40 70  --n_aug_support_samples 1  --penalty_c 1. --t_lr 1e-3 --num_workers 2 --idea_variant   --test --confusion --k_c 4  --n_shot 1 --n_symmetry_aug 1 --dataset tieredimagenet --model_id 1

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method good_metric  --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 100  --milestones 40 70 --k_gen 1   --confusion --k_c 16 --confusion_drop_rate 0 --dataset tieredimagenet --model_type last

#1 shot : 73.42  5 shot: 87.33
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method good_metric  --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 100  --milestones 40 70 --k_gen 2  --confusion --k_c 16 --confusion_drop_rate 0 --dataset tieredimagenet --model_type last

#python eval.py --gpu ${gpuid} --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_type best --method good_metric  --penalty_c  1. --idea_variant   --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1 --dataset tieredimagenet --confusion --distill_model  tieredimagenet_good_metric_resnet12_distill_1_gen_last.pth

