#训练文件
#常见学习率参考文档： https://www.zhihu.com/question/418989024
cliplr=1e-5
# ImageAdapter的学习率设置为5e-6，但是多次实验，并没有提升效果
basiclr=1e-4
#`weight_decay`的大小就是公式中的`λ`，可以理解为`λ`越大，优化器就越限制权重变得趋近 0
# 使用 weight decay 可以：
  #防止过拟合
  #- 保持权重在一个较小在的值，避免梯度爆炸。
  #- 因为在原本的 loss 函数上加上了权重值的 L2 范数，在每次迭代时，模不仅会去优化/最小化 loss，还会使模型权重最小化。
  #- 让权重值保持尽可能小，有利于控制权重值的变化幅度(如果梯度很大，说明模型本身在变化很大，去过拟合样本)，从而避免梯度爆炸
# 从实验结果上看，weight_decay最好不变
# weight_decay adds L2 regularization to the optimizer
#I have finetuned CLIP on PASCAL VOC2012 dataset using this implementation. When I changed the weight decay from 0.2 to 0.001, it works well on my task.
#https://github.com/openai/CLIP/issues/150
weight_decay=0.2
#betas are used for the optimization algorithm
beta1=0.9
#beta2=0.98
beta2=0.999
#eps is a small value to prevent division by zero
eps=1e-6
nepoch=30
batchsize=16
test_batchsize=16
dataset='OfficeHome'
test_envs=0
mode='ftAdapter'
#经验值设置为GPU数量的4倍，但是会增加CPU内存的开销
N_WORKERS=2
#optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from clip paper, the lr is smaller, more safe for fine tuning to new dataset

# 试一下同时调整视觉编码器和文本编码器
# lp模式作为baseline
#parser.add_argument('--ClipMode', type=str, default='Frozen',choices=["Forzen","Partial","Full"])#Clip模式选择
# 冻结Clip时，可以将lr=1e-3
#Clip全调的参数
#python main.py --mode $mode --ClipMode "Frozen" --dataset  $dataset --dduse --test_envs  0 --cliplr 1e-6 --basiclr 5e-6 --weight_decay $weight_decay  --beta1 $beta1 --beta2 $beta2 --eps $eps --nepoch $nepoch --batchsize $batchsize --test_batchsize $test_batchsize  --steps_per_epoch 220  --N_WORKERS 2
python main.py --mode $mode --ClipMode "Frozen" --dataset  $dataset  --test_envs  0 --cliplr 1e-6  --basiclr 1e-5 --weight_decay $weight_decay  --beta1 $beta1 --beta2 $beta2 --eps $eps --nepoch $nepoch --batchsize $batchsize --test_batchsize $test_batchsize  --steps_per_epoch 220  --N_WORKERS 2

#试试将Adapter修改为transformer
