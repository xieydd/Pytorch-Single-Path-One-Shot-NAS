python -m torch.distributed.launch --nproc_per_node=1 \
    ../train_imagenet.py --name=spos-supernet+-train-imagenet\
    --dataset=imagenet --gpus=0 --batch_size=256 --epochs=120 \
    --workers=30 --train_portion=1.0 --val_portion=0.01 \
    -a shuffleNas --w_lr=0.65 --w_weight_decay=0.00004 \
    --epoch-start-cs 0 --use-se --last-conv-after-pooling  --channels-layout OneShot \
    --label-smoothing --warmup-epochs 5
