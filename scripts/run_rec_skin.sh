gpuid=0
cd ../

#skin5/5
#59.36 +- 0.26
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_way 20  --n_symmetry_aug 1  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B --dataset skin198 --LR  --penalty_c 2.
#65.93
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_way 20  --n_symmetry_aug 1  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B --dataset skin198 --LR  --penalty_c 10.
##66.94
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_way 20  --n_symmetry_aug 1  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B --dataset skin198 --LR  --penalty_c 20.
##64.60
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0005 --test_times 1 --optim SGD --lr 5 --transform B --dataset skin198
##39.35
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 5 --transform B --dataset skin198
##65.71
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim SGD --lr 5 --transform B --dataset skin198

#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.00005 --test_times 1 --optim SGD --lr 5 --transform B --dataset skin198
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim SGD --lr 10 --transform B --dataset skin198
#70.36
#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim Adam --lr 0.005 --transform B_s --dataset skin198

#skin 3/7
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_way 20  --n_symmetry_aug 1  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B --dataset skin198 --LR  --penalty_c 20. --skin_split 0.3
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim Adam --lr 0.005 --transform B_s --dataset skin198 --skin_split 0.3

#skin 1/9
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 1 --n_way 20  --n_symmetry_aug 1  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.005 --test_times 1 --optim SGD --lr 0.5 --transform B --dataset skin198 --LR  --penalty_c 20. --skin_split 0.1
python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.0001 --test_times 1 --optim Adam --lr 0.005 --transform B_s --dataset skin198 --skin_split 0.1




#python eval.py --gpu ${gpuid}  --continue_pretrain --reduce_dim 128 --transform B --n_episodes 500 --model_id 1 --method confusion  --idea_variant  --n_aug_support_samples 5 --n_way 20  --n_symmetry_aug 5  --prompt --distill_model  ResNet12_stl_deepbdc_distill/last_model.tar --wd_test 0.001 --test_times 1 --optim SGD --lr 0.001 --transform B --dataset skin198
