# Model Card

## Model

- Architecture: DINOv2 ViT-S/14 encoder, DPT-lite decoder, shallow detail branch.
- Output: iris, visible eye region, pupil and occlusion-boundary supervision heads.
- Input: RGB image; training size 448 × 672 with global/native-detail dual views.
- Checkpoint: `models/iris_segmentation_ema_fp16.pth`.

## Training summary

- 4,433 samples from 12 sources.
- 12 epochs, effective batch size 8.
- Final training loss: 0.21809.
- EMA weights are exported for inference.

The private training images and manifests are not distributed.

## Limitations

- No independent subject-disjoint validation score is reported.
- Full-face `fast` inference can miss small irises.
- Multi-scale inference may produce false positives on hair, glasses, reflections or clothing.
- Performance can change with camera, illumination, ethnicity, gaze direction and eye size.
- This model is not validated for medical diagnosis, biometric authentication or safety-critical use.

## Intended use

Research, educational experiments, annotation assistance and transfer learning. Human review is required for generated masks.

## Privacy

Iris and face imagery is biometric data. Obtain consent, minimize retention, restrict access and avoid publishing identifiable samples without explicit permission.

