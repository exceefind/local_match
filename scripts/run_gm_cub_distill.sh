gpuid=6
cd ../../

python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 10 --test_LR --transform B --n_episodes 1000 --method good_metric --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 120 170  --n_aug_support_samples 1  --penalty_c 20. --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --test --confusion  --n_shot 1 --n_symmetry_aug 1 --dataset cub --model resnet18 --img_size 224 --k_c 4


echo "============= distill generation 1 ============="
python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method good_metric  --idea_variant --penalty_c .1 --n_aug_support_samples 1 --confusion --dataset cub --model resnet18 --img_size 224  --k_gen 1 --k_c 4 --max_epoch 220 --milestones 120 170 --penalty_c 20.  --model_type last

max=18
for i in `seq 2 $max`
do
echo "============= distill generation $i ============="
python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method good_metric  --idea_variant --penalty_c .1 --n_aug_support_samples 1 --confusion --dataset cub --model resnet18 --img_size 224  --k_gen $i --k_c 4 --max_epoch 220 --milestones 120 170 --penalty_c 20. --model_type last
done