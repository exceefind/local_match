cd ../

python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 5 --test_LR --transform B --n_episodes 1000 --model_id 4 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4  --gama 0.1 --test --n_shot 1 --constrastive
