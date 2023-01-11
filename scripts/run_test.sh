cd ../

#python eval.py --gpu 7 --continue_pretrain --reduce_dim 192 --transform B2 --n_episodes 1000 --model_id 29 --method good_metric  --learnable_alpha --penalty_c 1 --idea_variant --normalize_feat --FPN_list 2 3 --n_aug_support_samples 10 --n_shot 5
#
#python eval.py --gpu 7 --continue_pretrain --reduce_dim 192 --transform B --n_episodes 1000 --model_id 29 --method good_metric  --learnable_alpha --penalty_c 1 --idea_variant --normalize_feat --FPN_list 2 3 --n_aug_support_samples 10 --n_shot 5
#
#
#python eval.py  --gpu 7 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method good_metric --transform B --FPN_list 2 3 --distill_model  miniimagenet_good_metric_resnet12_distill_27_1_gen.pth --n_aug_support_samples 10 --FPN_list 2 3 --stop_grad  --learnable_alpha --reduce_dim 160 --normalize_feat

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 35 --method good_metric   --penalty_c .5 --idea_variant --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 8

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 35 --method good_metric   --penalty_c .1 --idea_variant --normalize_feat  --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 8

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 35 --method good_metric   --penalty_c .1 --idea_variant --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 8

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 35 --method good_metric   --penalty_c .1 --idea_variant --normalize_feat  --n_aug_support_samples 20 --n_shot 5 --confusion --k_c 8

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 36 --method good_metric   --penalty_c .5 --idea_variant --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 8 --FPN_list 2 3 --learnable_alpha

#39
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 39 --method good_metric   --penalty_c 2 --idea_variant  --normalize_feat  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 39 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 4
#
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 39 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 39 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 4

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 5 --n_shot 1 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c 1. --idea_variant  --normalize_feat  --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 20 --n_shot 1 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 40 --method good_metric   --penalty_c 1 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 8


#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 41 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 41 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 41 --method good_metric   --penalty_c 1 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 8
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c 1. --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 5 --n_shot 1 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c 1. --idea_variant  --normalize_feat  --n_aug_support_samples 5 --n_shot 1 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 5 --n_shot 1 --confusion --k_c 4

#
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 42 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 42 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4
#python eval.py --gpu 4 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_1_gen.pth --n_aug_support_samples 10 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c .5

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 38 --method good_metric   --penalty_c .5 --idea_variant --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 8 --FPN_list 2 3 --learnable_alpha
#
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 38 --method good_metric   --penalty_c .1 --idea_variant --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 8 --FPN_list 2 3 --learnable_alpha
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c .5 --idea_variant   --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 4 --symmetry_aug
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c 1. --idea_variant    --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c .5 --idea_variant    --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c .1 --idea_variant    --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4
#python eval.py  --gpu 0 --test_LR --val_freq 1   --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 1 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c .1 --n_symmetry_aug 1 --n_shot 1
#python eval.py  --gpu 0 --test_LR --val_freq 1   --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 1 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c 1. --n_symmetry_aug 1 --n_shot 1
#python eval.py  --gpu 0 --test_LR --val_freq 1   --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 1 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c 2.5 --n_symmetry_aug 1 --n_shot 1
#python eval.py  --gpu 0 --test_LR --val_freq 1   --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 1 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c .1 --n_symmetry_aug 1 --n_shot 5
#python eval.py  --gpu 0 --test_LR --val_freq 1   --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 5 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c 2. --n_symmetry_aug 1 --n_shot 5

#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 10 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c .1 --n_symmetry_aug 5 --num_workers 1

#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 10 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c .5 --n_symmetry_aug 10 --num_workers 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c 2. --idea_variant  --normalize_feat  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .1 --idea_variant  --normalize_feat  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 10

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c 2 --idea_variant   --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c .1 --idea_variant   --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric  --penalty_c .1 --idea_variant   --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 45 --method good_metric   --penalty_c 2. --idea_variant   --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 45 --method good_metric  --penalty_c .1 --idea_variant   --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 45 --method good_metric   --penalty_c .5 --idea_variant   --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 45 --method good_metric  --penalty_c .5 --idea_variant   --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 15 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 15
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c 2 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 10
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .5 --idea_variant  --normalize_feat  --n_aug_support_samples 20 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 20
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 43 --method good_metric   --penalty_c .05 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 10

#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 5 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c 1. --symmetry_aug

#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method good_metric --transform B  --distill_model  miniimagenet_good_metric_resnet12_distill_39_2_gen.pth --n_aug_support_samples 1 --stop_grad  --reduce_dim 160 --normalize_feat --confusion --penalty_c .1 --n_symmetry_aug 1 --num_workers 1 --dataset cub
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_distill_2_gen.pth --n_aug_support_samples 1 --dataset cub

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000  --method good_metric   --penalty_c .5 --idea_variant   --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 10 --dataset cub --model resnet18 --img_size 224
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000  --method good_metric   --penalty_c .5 --idea_variant   --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 10 --dataset cub --model resnet18 --img_size 224

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 46 --method good_metric  --penalty_c 2.5 --idea_variant   --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 46 --method good_metric  --penalty_c .1 --idea_variant   --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1
#
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 46 --method good_metric  --penalty_c .5 --idea_variant   --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 46 --method good_metric  --penalty_c .5 --idea_variant   --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
#
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 46 --method good_metric  --penalty_c .5 --idea_variant   --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 10
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 46 --method good_metric  --penalty_c .5 --idea_variant   --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 10

#47
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 47 --method good_metric  --penalty_c .1 --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 1 --confusion
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 47 --method good_metric  --penalty_c 2. --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 1 --confusion

#48
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 48 --method good_metric  --penalty_c .1 --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --confusion
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 48 --method good_metric  --penalty_c 2. --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --confusion

#49
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 49 --method good_metric  --penalty_c 2. --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --confusion
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 49 --method good_metric  --penalty_c 2. --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --confusion


#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 1 --penalty_c 2 --n_symmetry_aug 1
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 10 --penalty_c 0.1 --n_symmetry_aug 10
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 1 --penalty_c 2 --n_symmetry_aug 10

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000 --model_id 47 --method good_metric  --penalty_c 1. --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 1 --confusion
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c 2 --idea_variant   --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1

#python eval.py --gpu 5 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c 2 --idea_variant   --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1 --dataset tieredimagenet --confusion

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c .5 --idea_variant  --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c 2. --idea_variant  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 10

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 160 --transform B --n_episodes 1000 --model_id 39 --method good_metric   --penalty_c 2 --idea_variant  --normalize_feat  --n_aug_support_samples 10 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 10

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 56 --method good_metric   --penalty_c .5 --idea_variant  --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1 --temp_element

#deepbdc 采用 support 和 sym_aug 的性能表现：
## 1shot
##65.21
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 5 --penalty_c .1 --n_symmetry_aug 5
##64.19
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 5 --penalty_c .5 --n_symmetry_aug 5
##63.53
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 5 --penalty_c 1. --n_symmetry_aug 5
##66.12
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 10 --penalty_c .1 --n_symmetry_aug 10
##64.91
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 10 --penalty_c .5 --n_symmetry_aug 10
##64.44
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 1 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 10 --penalty_c 1. --n_symmetry_aug 10
#
##84.53
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 5 --penalty_c .5 --n_symmetry_aug 5
##84.32
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 5 --penalty_c 1. --n_symmetry_aug 5
##84.27
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 5 --penalty_c 2 --n_symmetry_aug 5
##85.26
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 10 --penalty_c .5 --n_symmetry_aug 10
##85.19
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 10 --penalty_c 1. --n_symmetry_aug 10
##85.09
#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method stl_deepbdc --transform B --distill_model  miniimagenet_stl_deepbdc_resnet12_best.pth --n_aug_support_samples 10 --penalty_c 2 --n_symmetry_aug 10

#58
#65.09
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 58 --method good_metric  --penalty_c .5 --idea_variant  --n_aug_support_samples 5 --n_shot 1 --confusion --k_c 2 --n_symmetry_aug 1
#65.15
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 58 --method good_metric  --penalty_c .1 --idea_variant  --n_aug_support_samples 5 --n_shot 1 --confusion --k_c 2 --n_symmetry_aug 1
#82.08
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 58 --method good_metric  --penalty_c .5 --idea_variant  --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1
#81.95
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 58 --method good_metric  --penalty_c 1.5 --idea_variant  --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1
#82
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 58 --method good_metric  --penalty_c 2.5 --idea_variant  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1

#64.77  +normalize
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 58 --method good_metric  --penalty_c .1 --idea_variant  --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 2 --n_symmetry_aug 1 --normalize_feat

#65.35
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 58 --method good_metric  --penalty_c .1 --idea_variant  --n_aug_support_samples 1 --n_shot 1 --confusion --k_c 2 --n_symmetry_aug 1 --normalize_feat

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 58 --method good_metric  --penalty_c .1 --idea_variant  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1 --normalize_feat

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 44 --method good_metric   --penalty_c 2. --idea_variant   --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 4 --n_symmetry_aug 1
python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 66 --method good_metric  --penalty_c .5 --idea_variant  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 67 --method good_metric  --penalty_c 2 --idea_variant  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 68 --method good_metric  --penalty_c 2 --idea_variant  --n_aug_support_samples 1 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1


#python eval.py --gpu 0 --continue_pretrain --reduce_dim 36 --transform B --n_episodes 1000 --model_id 57 --method good_metric  --penalty_c 1. --idea_variant  --n_aug_support_samples 5 --n_shot 5 --confusion --k_c 2 --n_symmetry_aug 1

#python eval.py  --gpu 0 --test_LR --val_freq 1  --n_shot 5 --n_episodes 1000  --continue_pretrain --method good_metric --transform B --distill_model  cub_good_metric_resnet12_distill_8_gen_best.pth --n_aug_support_samples 1 --penalty_c 20 --n_symmetry_aug 1 --idea_variant --confusion --dataset cub --model resnet18 --img_size 224
#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000  --method good_metric   --penalty_c 20. --idea_variant   --n_aug_support_samples 1 --distill_model  cub_good_metric_resnet12_distill_8_gen_best.pth --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 1 --dataset cub --model resnet18 --img_size 224

#python eval.py --gpu 0 --continue_pretrain --reduce_dim 196 --transform B --n_episodes 1000  --method good_metric   --penalty_c .1 --idea_variant   --n_aug_support_samples 10 --n_shot 1 --confusion --k_c 4 --n_symmetry_aug 10 --dataset cub --model resnet18 --img_size 224


