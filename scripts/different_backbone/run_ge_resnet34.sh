gpuid=0

cd ../../

python train_pretrain.py --gpu ${gpuid}  --test_LR --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 2000  --transform B --n_episodes 1000 --model_id 1002 --method confusion --batch_size 64 --reduce_dim 640 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 1  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --n_symmetry_aug 1 --embeding_way GE --LR --my_model --model resnet34 --img_size 224

python distillation.py --gpu ${gpuid} --test_LR --val_freq 2000  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 640 --transform B  --method confusion  --model_id 1002 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 170  --milestones 100 150 --k_gen 1  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --embeding_way GE --LR --my_model --model_type last --cross_all --model resnet34  --img_size 224

python distillation.py --gpu ${gpuid} --test_LR --val_freq 2000  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 640 --transform B  --method confusion  --model_id 1002 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 170  --milestones 100 150 --k_gen 2  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --embeding_way GE --LR --my_model --model_type last --cross_all --model resnet34  --img_size 224