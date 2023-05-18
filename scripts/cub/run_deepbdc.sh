gpuid=0
cd ../../

#python train_pretrain.py  --gpu ${gpuid}  --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 10 --test_LR --n_aug_support_samples 1 --transform B --n_episodes 1000 --method stl_deepbdc --batch_size 64 --gama 0.1 --milestones 120 170 --t_lr 1e-3 --penalty_c 20. --dataset cub --model resnet18 --img_size 224 --test

#echo "============= distill generation 1 ============="
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method good_metric  --idea_variant --penalty_c .1 --n_aug_support_samples 1 --confusion --dataset cub --model resnet18 --img_size 224  --k_gen 1 --k_c 4 --max_epoch 220 --milestones 120 170 --penalty_c 20.  --model_type last
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method stl_deepbdc --k_gen 1 --penalty_c 20. --model_type last --dataset cub --model resnet18 --img_size 224  --k_gen 1 --k_c 4 --max_epoch 220 --milestones 120 170

#max=18
#for i in `seq 2 $max`
#do
#echo "============= distill generation $i ============="
##python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method good_metric  --idea_variant --penalty_c .1 --n_aug_support_samples 1 --confusion --dataset cub --model resnet18 --img_size 224  --k_gen $i --k_c 4 --max_epoch 220 --milestones 120 170 --penalty_c 20. --model_type last
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method stl_deepbdc --k_gen 1 --penalty_c 20. --model_type last --dataset cub --model resnet18 --img_size 224  --k_gen $i --k_c 4 --max_epoch 220 --milestones 120 170
#done

#python eval.py  --gpu ${gpuid} --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  cub_stl_deepbdc_resnet12_distill_18_gen_last.pth --n_aug_support_samples 1 --penalty_c 20. --dataset cub --model resnet18 --img_size 224
#python eval.py  --gpu ${gpuid} --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  cub_stl_deepbdc_resnet12_distill_18_gen_last.pth --n_aug_support_samples 1 --penalty_c 20. --dataset cub --model resnet18 --img_size 224

python eval.py  --gpu ${gpuid} --test_LR --val_freq 1  --n_shot 1 --n_episodes 2000  --continue_pretrain --method stl_deepbdc --transform B  --n_aug_support_samples 1 --penalty_c 20. --dataset cub --model resnet18 --img_size 224 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --seed 0 --test_times 5
#
#python eval.py  --gpu ${gpuid} --test_LR --val_freq 1  --n_shot 5 --n_episodes 2000  --continue_pretrain --method stl_deepbdc --transform B  --n_aug_support_samples 1 --penalty_c 20. --dataset cub --model resnet18 --img_size 224 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --seed 0 --test_times 5
