gpuid=1

cd ../../

#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 220 --val_freq 500  --transform B --n_episodes 100 --model_id 21 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 120 170  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 2 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 5 --dataset cub --model resnet18 --img_size 224
##
#max=18
#for i in `seq 11 $max`
#do
#echo "============= distill generation $i ============="
##python distillation.py --gpu ${gpuid} --test_LR --val_freq 500  --n_shot 1 --n_episodes 500 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 21 --idea_variant  --n_aug_support_samples 5 --max_epoch 220  --milestones 120 170 --k_gen $i  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --dataset cub --model resnet18 --img_size 224 --model_type last --wd_test 1e-3
#done
##


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-4 --distill_model cub_confusion_resnet12_distill_21_12_gen_last.pth --penalty_c 20. --prompt --LR
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-4 --distill_model cub_confusion_resnet12_distill_21_4_gen_last.pth --penalty_c 20. --prompt


#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 1 --model_type best --prompt



#Good Embedding
#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 20  --transform B --n_episodes 1000 --model_id 10 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 5 --embeding_way GE

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 20  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 10 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 170  --milestones 100 150 --k_gen 1  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --embeding_way GE
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 10 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --embeding_way GE


#86.09
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-4 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar  --penalty_c 20 --prompt --LR --test_times 5
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-4 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar  --penalty_c 20 --prompt --LR --test_times 5


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-4 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --penalty_c 20. --prompt  --optim SGD --lr 0.1

#new
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-3 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --penalty_c 20. --prompt  --optim SGD --lr 0.5 --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --n_episodes 2000 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-3 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --penalty_c 20. --prompt --optim SGD --lr 0.1 --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 100 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-3 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --penalty_c 20. --prompt  --optim SGD --lr 0.5 --test_times 1 --transform B_s
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 100 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-3 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --penalty_c 20. --prompt  --optim SGD --lr 0.5 --test_times 1 --transform B_s
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 500 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 5e-4 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --penalty_c 20. --prompt  --optim SGD --lr 0.5 --test_times 1 --transform B_s
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 500 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 5e-4 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --penalty_c 20. --prompt  --optim SGD --lr 0.5 --test_times 1 --transform B_s

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 21 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last  --dataset cub --model resnet18 --img_size 224 --wd_test 1e-3 --distill_model ResNet18_stl_deepbdc_distill_cub/last_model.tar --penalty_c 20. --prompt --optim SGD --lr 0.1

