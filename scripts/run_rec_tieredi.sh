gpuid=0

cd ../

#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 100 --val_freq 50  --transform B --n_episodes 1000 --model_id 1 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 40 70  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 5 --dataset tieredimagenet

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 100  --milestones 40 70 --k_gen 1  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --dataset tieredimagenet --model_type last

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 100  --milestones 40 70 --k_gen 2  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --dataset tieredimagenet --model_type last

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 100  --milestones 40 70 --k_gen 3  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --dataset tieredimagenet --model_type last --wd_test 1e-3

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 10  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 1 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 100  --milestones 40 70 --k_gen 4  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --dataset tieredimagenet --model_type last --wd_test 1e-3


#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth --wd_test 1e-3

python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 1e-3

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 1e-4
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 100 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --distill_model tieredimagenet_confusion_resnet12_distill_1_2_gen_last.pth  --wd_test 1e-3

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --dataset tieredimagenet --model_type last


#python eval.py --gpu 0 --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 3 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 1 --model_type best --prompt



#Good Embedding
#python train_pretrain.py --gpu ${gpuid} --MultiStepLr --lr 0.05 --max_epoch 170 --val_freq 20  --transform B --n_episodes 1000 --model_id 10 --method confusion --batch_size 64 --reduce_dim 128 --drop_rate 0.5 --milestones 100 150  --n_aug_support_samples 5  --penalty_c .1 --t_lr 1e-3 --num_workers 4 --idea_variant  --gama 0.1 --n_shot 1  --confusion --prompt --test --n_symmetry_aug 5 --embeding_way GE

#python distillation.py --gpu ${gpuid} --test_LR --val_freq 20  --n_shot 1 --n_episodes 1000 --t_lr 1e-3 --reduce_dim 128 --transform B  --method confusion  --model_id 10 --idea_variant  --penalty_c .1 --n_aug_support_samples 5 --max_epoch 170  --milestones 100 150 --k_gen 1  --confusion --prompt --n_symmetry_aug 5 --num_workers 2 --embeding_way GE
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 1000 --model_id 10 --method confusion  --idea_variant  --n_aug_support_samples 10 --n_shot 5  --n_symmetry_aug 10 --model_type last --prompt --embeding_way GE


