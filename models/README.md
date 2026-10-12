# Inference checkpoint

`iris_segmentation_ema_fp16.pth` contains the epoch-12 EMA state dict stored in FP16 plus minimal model metadata.

It intentionally excludes optimizer, scheduler, gradient scaler, raw training state, sample paths and sample hashes. Loading converts tensors into the model's runtime dtype. Use the local full `last.pth` checkpoint when resuming training.

