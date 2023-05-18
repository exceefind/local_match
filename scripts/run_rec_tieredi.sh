gpuid=0

cd ../

#采用LR：

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.01  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s --LR_rec --penalty_c 2.



#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 10  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.01  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s --drop_few 0.5 --LR --penalty_c 10

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 10  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.0005  --optim Adam --lr 0.001 --my_model --test_times 1 --transform B_s --drop_few 0.5 --LR_rec --penalty_c 10
#


#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 100 --val_freq 50  --transform B --n_episodes 1000 --model_id 1 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 40 70  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 5 --dataset tieredimagenet

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 100  --milestones 40 70 --k_gen 1  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --dataset tieredimagenet --model_type last --LR

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 100  --milestones 40 70 --k_gen 2  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --dataset tieredimagenet --model_type last --LR
#
#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 100  --milestones 40 70 --k_gen 3  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --dataset tieredimagenet --model_type last --LR

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 1 --max_epoch 100  --milestones 40 70 --k_gen 4  --confusion --prompt --n_symmetry_aug 1 --num_workers 2 --dataset tieredimagenet --model_type last --LR

#BDC LR test
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth --wd_test 1e-3 --LR  --penalty_c 0.1 --test_times 1 --my_model
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth --wd_test 1e-3 --LR  --penalty_c 2.0 --test_times 1 --my_model
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_3_gen_last.pth --wd_test 1e-3 --LR  --penalty_c 0.1 --test_times 1 --my_model
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_3_gen_last.pth --wd_test 1e-3 --LR  --penalty_c 2.0 --test_times 1 --my_model
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_4_gen_last.pth --wd_test 1e-3 --LR  --penalty_c 0.1 --test_times 1 --my_model
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_4_gen_last.pth --wd_test 1e-3 --LR  --penalty_c 2.0 --test_times 1 --my_model

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_3_gen_last.pth --wd_test 1e-3 --LR  --penalty_c 0.1 --test_times 1 --my_model
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_3_gen_last.pth --wd_test 1e-3 --LR  --penalty_c 2.0 --test_times 1 --my_model


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 1e-3

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 1e-4
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 1e-3

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --model_type last


#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 1 --model_type best --prompt


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_4_gen_last.pth  --wd_test 0.1 --optim SGD --lr 0.5 --my_model --test_times 1
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_4_gen_last.pth  --wd_test 0.01 --optim SGD --lr 0.5 --my_model --test_times 1


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001 --optim Adam --lr 0.005 --my_model --test_times 1 --transform B

#new 74.48
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.2 --optim SGD --lr 0.5 --my_model --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1 --optim SGD --lr 0.5 --my_model --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 20 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1 --optim SGD --lr 0.5 --my_model --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1 --optim SGD --lr 0.5 --my_model --test_times 1 --transform B

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1 --optim SGD --lr 0.1 --my_model --test_times 1 --transform B

#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1 --optim SGD --lr 0.1 --my_model --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.0001 --optim Adam --lr 0.001 --my_model --test_times 1 --transform B_s
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1 --optim SGD --lr 0.1 --my_model --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1 --optim SGD --lr 0.1 --my_model --test_times 1 --transform B


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.01 --optim SGD --lr 0.5 --my_model --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001 --optim SGD --lr 0.5 --my_model --test_times 1 --transform B

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.0001 --optim Adam --lr 0.005 --my_model --test_times 1 --transform B

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.01 --optim SGD --lr 0.1 --my_model --test_times 1 --transform B


#new
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.0005  --optim SGD --lr 1 --my_model --test_times 1
#86.93
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 5  --n_symmetry_aug 10 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001  --optim Adam --lr 0.001 --my_model --test_times 1
#LR BDC --> 87.79
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001  --optim SGD --lr 0.5 --my_model --test_times 1 --LR --penalty_c 2

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001  --optim Adam --lr 0.005 --my_model --test_times 1
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 20 --n_shot 5  --n_symmetry_aug 20 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001  --optim Adam --lr 0.005 --my_model --test_times 1
##88.73
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.005  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B
##88.54
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B
##88.38
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.05  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B
##88.96
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.005  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s
##88.75
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s
##88.58
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.05  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s
#89.20
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.01  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s
#200:   0.7drop+0.05wd: 72.69 --> 0.1drop+0.1wd:72.84 -- 0.1drop+0.1wd+B:72.36-->0.2drop+0.1wd:72.75-->0.2drop+0.2wd:72.56---0.5drop+0.1wd: 72.82--0.5drop+0.1wd: 72.76----20aug +0.5drop +0.1wd: 72.93

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s --drop_few 0.5

#bast _new
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.05  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s --drop_few 0.5 --LR_rec  --penalty_c .1

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.01  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s --drop_few 0.5 --LR_rec  --penalty_c 2.


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.05  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s --drop_few 0.5
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.05  --optim SGD --lr 0.5 --my_model --test_times 3 --transform B_s --drop_few 0.5 --LR_rec --penalty_c 0.1
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.05  --optim Adam --lr 0.005 --my_model --test_times 3 --transform B_s --drop_few 0.5
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.05  --optim SGD --lr 0.5 --my_model --test_times 3 --transform B_s --drop_few 0.5 --LR_rec --penalty_c 0.1


#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 10 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.1  --optim SGD --lr 2 --my_model --test_times 1 --transform B_s --drop_few 0.5
#
##5shot :10aug :88.85
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 5  --n_symmetry_aug 10 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.01  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s


#LR : 88.76

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.005  --optim SGD --lr 0.5 --my_model --test_times 1 --transform B_s --LR --penalty_c 2.



#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 5  --n_symmetry_aug 10 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.0001  --optim SGD --lr 0.5 --my_model --test_times 1  --LR --penalty_c 2



#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.01  --optim Adam --lr 0.001 --my_model --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.001  --optim Adam --lr 0.001 --my_model --test_times 1 --transform B
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 0.0001  --optim Adam --lr 0.001 --my_model --test_times 1 --transform B


#Good Embedding

#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 20  --transform B --n_episodes 1000 --model_id 10 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 5 --embeding_way GE

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 20  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 10 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 170  --milestones 100 150 --k_gen 1  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --embeding_way GE
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 10 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 5  --n_symmetry_aug 10 --model_type last --prompt --embeding_way GE


