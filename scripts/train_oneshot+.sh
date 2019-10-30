python -m torch.distributed.launch --nproc_per_node=1 ../train_imagenet.py
    --name=spos-oneshot+-train-imagenet \
    --dataset=imagenet --gpus=0
    --batch_size=250 --epochs=360 --workers=30 \
    --train_portion=1.0 --val_portion=0.01 \
    -a shuffleNas_fixArch --label-smoothing \
    --w_lr=1.3 --w_weight_decay=0.00003 --warmup-epochs 5 
    --block-choices '0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2' \
    --channel-choices '6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3' \
    --channels-layout OneShot --use-se --last-conv-after-pooling \
