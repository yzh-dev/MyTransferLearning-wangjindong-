import os

cmd0 = "python train.py --algorithm ERM --dataset office-home --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"
cmd1 = "python train.py --algorithm DANN --dataset office-home --alpha 0.1 --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"
cmd2 = "python train.py --algorithm Mixup --dataset office-home --mixupalpha 0.2 --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"
cmd3 = "python train.py --algorithm RSC --dataset office-home --rsc_f_drop_factor 0.1 --rsc_b_drop_factor 0.1 --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"
cmd4 = "python train.py --algorithm MMD --dataset office-home --mmd_gamma 1 --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"
cmd5 = "python train.py --algorithm CORAL --dataset office-home --mmd_gamma 1 --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"
cmd6 = "python train.py --algorithm ANDMask --dataset office-home --tau 0.3 --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"
cmd7 = "python train.py --algorithm ANDMask --dataset office-home --tau 0.5 --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"
cmd8 = "python train.py --algorithm VREx --dataset office-home --lam 100 --anneal_iters 100  --lr 5e-4 --max_epoch 60 --batch_size 32 --test_envs 1"

os.system(cmd0)
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)
os.system(cmd5)
os.system(cmd6)
os.system(cmd7)
os.system(cmd8)
