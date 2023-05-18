gpuid=0

cd ../../

python train_pretrain.py --gpu ${gpuid}  --test_LR --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 5  --transform B --n_episodes 1000 --model_id 3001 --method confusion --batch_size 64 --reduce_dim 640 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --n_shot 5  --confusion --prompt --n_symmetry_aug 1 --embeding_way baseline++  --my_model --wd_test 0.01
#
#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 100 --val_freq 2000  --transform B --n_episodes 1000 --model_id 3001 --method confusion --batch_size 64 --reduce_dim 640 --drop_rate 0.5 --milestones 40 70  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt  --n_symmetry_aug 1 --dataset tieredimagenet --n_symmetry_aug 1 --embeding_way baseline++ --LR --my_model

#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 10  --transform B --n_episodes 100 --model_id 3001 --method confusion --batch_size 64 --reduce_dim 512 --drop_rate 0.5 --milestones 120 170  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 1 --dataset cub --model resnet18 --embeding_way baseline++ --img_size 224 --my_model --wd_test 0.001
