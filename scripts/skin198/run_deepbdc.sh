gpuid=0
cd ../../

#python train_pretrain.py  --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 50 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --method stl_deepbdc --batch_size 64 --gama 0.1 --penalty_c 20. --milestones 120 170 --t_lr 1e-3 --dataset skin198 --model_id 0 --test

python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --n_aug_support_samples 1 --penalty_c 20. --n_symmetry_aug 1 --dataset skin198 --model_id 0
