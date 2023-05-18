gpuid=0

cd ../

#sigmoid
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim SGD --lr 0.5

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --distill_model ResNet12_good_embed_distil/last_model.tar --embeding_way GE  --LR --penalty_c 1
#pc=2 ：80.85
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --distill_model ResNet12_good_embed_distil/last_model.tar  --wd_test 0.1 --optim SGD --lr 0.5 --embeding_way GE --transform B

#64.08
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --distill_model ResNet12_good_embed_distil/last_model.tar  --wd_test 0.1 --optim SGD --lr 1 --embeding_way GE --transform B_s --penalty_c 0.1 --LR
#80.82
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --distill_model ResNet12_good_embed_distil/last_model.tar  --wd_test 0.1 --optim SGD --lr 1 --embeding_way GE --transform B_s --penalty_c 2 --LR
#
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 1 --model_type last --prompt --distill_model ResNet12_good_embed_distil/last_model.tar  --wd_test 0.1 --optim SGD --lr 1 --embeding_way GE --transform B_s --penalty_c 1 --LR
#
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 1 --model_type last --prompt --distill_model ResNet12_good_embed_distil/last_model.tar  --wd_test 0.1 --optim SGD --lr 1 --embeding_way GE --transform B_s --penalty_c 1 --LR

#82.71 softmax
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5 --model_type last --prompt --distill_model ResNet12_good_embed_distil/last_model.tar  --wd_test 0.001 --optim SGD --lr 0.5 --embeding_way GE --transform B_s --penalty_c 1.
#65.97
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 640  --n_episodes 2000 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5 --model_type last --prompt --distill_model ResNet12_good_embed_distil/last_model.tar  --wd_test 0.1 --optim SGD --lr 0.5 --embeding_way GE --transform B_s



#softmax
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.5
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 1 --optim SGD --lr 0.5
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.1 --test_times 1 --optim SGD --lr 0.1
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 1  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.01 --test_times 1 --optim SGD --lr 0.1
#
#
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 200 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_shot 5  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim SGD --lr 0.5



