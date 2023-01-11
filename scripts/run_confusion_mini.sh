cd ../

#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 10 --test_LR --transform B --n_episodes 1000 --model_id 1 --method confusion --batch_size 64 --reduce_dim 256 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --test --n_shot 1  --confusion --confusion_drop_rate 0.1  --k_c 4

#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 5 --test_LR --transform B --n_episodes 1000 --model_id 3 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4  --gama 0.1 --test --n_shot 1

#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 5 --test_LR --transform B --n_episodes 1000 --model_id 3 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150 180  --n_aug_support_samples 1  --penalty_c 2. --t_lr 1e-3 --num_workers 4  --gama 0.1 --test --n_shot 5 --confusion --k_c 5

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 2 --method confusion  --penalty_c .1 --idea_variant  --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 2 --n_symmetry_aug 1 --model_type last

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 2 --method confusion  --penalty_c .5 --idea_variant  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1 --model_type last

python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --penalty_c 2.0 --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type best

python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --penalty_c 0.1 --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type best
