gpuid=0

cd ../

#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 1 --test_LR --transform B --n_episodes 1000 --model_id 1 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --LR
#
#python eval.py --gpu 3 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt
#--distill_model  miniimagenet_good_metric_resnet12_best_44.pth

#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 100  --transform B --n_episodes 1000 --model_id 1 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 1
#

python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 1  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 2  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 3  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 4  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 200  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 170  --milestones 100 150 --k_gen 5  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --model_type last
#

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --wd_test 0.05 --LR --penalty_c 2. --test_times 1 --distill_model miniimagenet_confusion_resnet12_distill_1_2_gen_last.pth

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --distill_model miniimagenet_confusion_resnet12_distill_1_3_gen_last.pth --wd_test 0.05


#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 1 --model_type best --prompt



#Good Embedding
#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 20  --transform B --n_episodes 1000 --model_id 11 --method confusion --batch_size 64 --reduce_dim 640 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 1 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 5 --embeding_way GE

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 20  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 10 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 170  --milestones 100 150 --k_gen 1  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --embeding_way GE
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640 --transform B --n_episodes 1000 --model_id 11 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1 --n_symmetry_aug 5 --model_type last --prompt --embeding_way GE

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640 --transform B --n_episodes 100 --model_id 11 --method confusion  --idea_variant  --n_aug_support_samples 20 --n_shot 5 --n_symmetry_aug 5 --model_type last --prompt --embeding_way GE



