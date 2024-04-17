lr=1e-5
weight_decay=0.2
beta1=0.9
beta2=0.98
eps=1e-6
nepoch=30
batchsize=16
test_batchsize=16
# officehome
dataset=0
test_data=1
mode='ft'
N_WORKERS=2

python main.py --mode $mode --dataset  $dataset  --test_data  $test_data --lr 1e-5  --weight_decay $weight_decay  --beta1 $beta1 --beta2 $beta2 --eps $eps --nepoch $nepoch --batchsize $batchsize --test_batchsize $test_batchsize

