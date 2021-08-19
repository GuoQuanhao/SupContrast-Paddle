# SupContrast-Paddle
```python
pip install tensorboard_logger

python main_supcon.py --batch_size 512 --learning_rate 0.5 --temp 0.1 --cosine

python main_linear.py --batch_size 512 --learning_rate 5 --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pdparmas
```

论文所达到精度如下
Results on CIFAR-10:
|          |Arch | Setting | Loss | Accuracy(%) |
|----------|:----:|:---:|:---:|:---:|
|  SupCrossEntropy | ResNet50 | Supervised   | Cross Entropy |  95.0  |
|  SupContrast     | ResNet50 | Supervised   | Contrastive   |  96.0  | 
|  SimCLR          | ResNet50 | Unsupervised | Contrastive   |  93.6  |

Paddle复现精度：95.94
