#python train_pretrain.py --gpu 2 --val_freq 1 --transform A --include_bg --beta 0.1 --local_mode cell_mix

#python train_pretrain.py --gpu 5 --MultiStepLr --lr 0.1 --max_epoch 150 --val_freq 1   --test_LR --n_aug_support_samples 5 --model_id 1 --transform A  --drop_gama 0.8

#python train_pretrain.py --gpu 1 --MultiStepLr --lr 0.1 --max_epoch 120 --val_freq 1 --include_bg  --beta 0.01 --model_id 5 --feature_pyramid --n_episodes 200

#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.1 --max_epoch 150 --val_freq 1 --include_bg  --beta 0.05 --model_id 4 --feature_pyramid --n_episodes 200 --transform B


#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.1 --max_epoch 120 --val_freq 1  --n_shot 1 --n_episodes 100  --model_id 2 --transform A --local_mode cell_mix --include_bg

#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.1 --max_epoch 120 --val_freq 1  --n_shot 1 --n_episodes 100  --model_id 3 --transform A --local_mode cell_mix --include_bg  --drop_gama 0.5

#python train_pretrain.py --gpu 6  --MultiStepLr --lr 0.1 --max_epoch 120 --val_freq 1 --include_bg --local_mode cell --no_save_model

#python train_pretrain.py --gpu 7 --MultiStepLr --lr 0.05 --max_epoch 150 --val_freq 1 --test_LR --n_aug_support_samples 5 --transform B --n_episodes 200 --model_id 1 --gama 0.1 --drop_gama 0.8 --transform B
#python train_pretrain.py --gpu 2 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 5 --transform B --n_episodes 200 --model_id 6 --gama 0.1 --drop_gama 1 --transform B
#python train_pretrain.py --gpu 3 --MultiStepLr --lr 0.002 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 5 --transform B --n_episodes 200 --model_id 10 --method good_metric --batch_size 64

#更大学习率  test_LR + ab + torch norm + abs
#python train_pretrain.py --gpu 0 --MultiStepLr --lr 0.05 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 5 --transform B --n_episodes 100 --model_id 12 --method good_metric --batch_size 64 --gama 0.1

#  ab +norm 2
python train_pretrain.py --gpu 4 --MultiStepLr --lr 0.5 --max_epoch 180 --val_freq 1 --test_LR --n_aug_support_samples 5 --transform B --n_episodes 100 --model_id 13 --method good_metric --batch_size 64 --gama 0.1