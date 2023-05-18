gpuid=0

cd ../../

#python train_pretrain.py  --gpu ${gpuid}  --MultiStepLr --lr 0.05 --max_epoch 220 --reduce_dim 512 --val_freq 50 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --method confusion --batch_size 64 --gama 0.1 --milestones 120 170 --t_lr 1e-3 --penalty_c 20. --dataset skin198 --model resnet18 --img_size 224 --test --embeding_way GE --LR --my_model --n_way 20 --model_id 1 --prompt

max=12
for i in `seq 3 $max`
do
echo "============= distill generation $i ============="
python distillation.py --gpu ${gpuid} --test_LR --val_freq 1000  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 512 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 220 --milestones 120 170 --k_gen $i  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --dataset skin198 --model_type last --LR --embeding_way GE --my_model --model resnet18 --img_size 224 --penalty_c 20. --n_way 20 --prompt
done
