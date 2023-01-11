
#python eval.py --gpu 1 --val_freq 1  --test_LR --n_aug_support_samples 5 --model_id 1 --transform A  --drop_gama 0.8 --continue_pretrain --n_shot 5


#python eval.py --gpu 0 --val_freq 1  --n_shot 1 --n_episodes 1000 --test_LR --n_aug_support_samples 5 --model_id 1 --continue_pretrain --drop_gama 0.8

python eval.py --gpu 0 --val_freq 1  --n_shot 1  --n_episodes 1000  --continue_pretrain --drop_gama 0.5  --feature_pyramid --include_bg  --model_id 4 --transform B
#python eval.py --gpu 1 --val_freq 1  --n_shot 1  --n_episodes 100  --continue_pretrain --drop_gama 0.5 --include_bg --temperature 6 --n_aug_support_samples 5 --n_local_proto 4
#python eval.py --gpu 1 --val_freq 1  --n_shot 1  --n_episodes 100  --continue_pretrain --drop_gama 0.5 --include_bg --temperature 12 --n_aug_support_samples 5
#python eval.py --gpu 1 --val_freq 1  --n_shot 1  --n_episodes 100  --continue_pretrain --drop_gama 0.5 --include_bg --temperature 18 --n_aug_support_samples 5
#python eval.py --gpu 1 --val_freq 1  --n_shot 1  --n_episodes 100  --continue_pretrain --drop_gama 0.5 --include_bg --temperature 25 --n_aug_support_samples 5

python eval.py --gpu 0 --val_freq 1  --n_shot 1  --n_episodes 1000  --continue_pretrain --drop_gama 1 --feature_pyramid --include_bg  --model_id 6 --transform B


