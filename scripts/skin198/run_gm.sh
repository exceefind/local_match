gpuid=6
cd ../../
# val: 70.98
#test_last : 66.47
#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 100 --test_LR --transform B --n_episodes 1000 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 120 170  --n_aug_support_samples 1 --stop_grad --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion --k_c 4 --confusion_drop_rate 0 --dataset skin198
#
#python eval.py --gpu ${gpuid} --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000  --method good_metric   --penalty_c 20. --idea_variant  --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1 --dataset skin198

python eval.py --gpu ${gpuid} --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000  --method good_metric   --penalty_c 1. --idea_variant  --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1 --dataset skin198

