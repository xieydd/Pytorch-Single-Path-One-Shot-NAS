python -m torch.distributed.launch --nproc_per_node=1 ../train_imagenet.py
    --name=spos-oneshot-s+-train-imagenet \
    --dataset=imagenet --gpus=0
    --batch_size=250 --epochs=360 --workers=30 \
    --train_portion=1.0 --val_portion=0.01 \
    -a shuffleNas_fixArch --label-smoothing \
    --w_lr=1.3 --w_weight_decay=0.00003 --warmup-epochs 5 \
    --block-choices '0, 0, 0, 1, 0, 0, 1, 0, 3, 2, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0' \
    --channel-choices '8, 7, 6, 8, 5, 7, 3, 4, 2, 4, 2, 3, 4, 5, 6, 6, 3, 3, 4, 6' \
    --channels-layout OneShot --use-se --last-conv-after-pooling \
