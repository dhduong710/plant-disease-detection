# Plant Disease Detection 

Deployment link of the project: https://inosuke710-plantcare.hf.space/

Download dataset here: https://www.kaggle.com/datasets/emmarex/plantdisease

This project focuses on building deep learning models for **plant disease detection** from leaf images. The dataset consists of ~20,000 images across multiple classes of healthy and diseased leaves.

I experimented with different CNN architectures and training strategies, then compared their performance on two test settings:

- **Normal Test**: Clean test set with standard preprocessing.
- **Hard Test**: Same test set but with strong augmentations (random crop, rotation, heavy color jitter). This setting evaluates robustness to domain shift.

---

## Model Performance

| Model                                   | Training Strategy                           | Normal Test Accuracy  | Hard Test Accuracy |
|-----------------------------------------|---------------------------------------------|-----------------------|--------------------|
| **Simple CNN (baseline)**               | 2 conv layers, weighted sampler             | 0.8849                | 0.3907             |
| **ResNet50 (fine-tuned)**               | Focal Loss, last 20 layers fine-tuned       | 0.9648                | 0.0963             |
| **MobileNetV2 (fine-tuned)**            | Standard fine-tuning                        | 0.9697                | 0.1676             |
| **EfficientNetB3 + CBAM + Mixup+CutMix**| Attention + advanced augmentation strategies| **0.9740**            | **0.9350**         |

---

## Analysis

- **Simple CNN (baseline)**: Performed decently on Normal Test (88.5%), but struggled under Hard Test (39.1%). Serves as a useful reference point to highlight the value of transfer learning and advanced architectures.
- **ResNet50**: Achieved high accuracy on clean data (96.5%), but collapsed on Hard Test (9.6%). Indicates overfitting and poor robustness.
- **MobileNetV2**: Similar trend as ResNet50. Despite being lightweight and efficient, its robustness was still weak (16.8% on Hard Test).
- **EfficientNetB3 + CBAM + Mixup+CutMix**: Not only achieved the **highest accuracy on clean data (97.4%)**, but also maintained **very high robustness (93.5%)** under strong augmentations.

---

## Conclusion

- **Baseline CNN** shows the limitations of shallow models without transfer learning.
- **ResNet50 / MobileNetV2** demonstrate strong performance on clean test data but fail to generalize under domain shift.
- **EfficientNetB3 with CBAM and advanced regularization (Mixup, CutMix)** is the clear winner, combining **state-of-the-art accuracy** and **robustness to real-world variations**.
- For deployment and real-world use, I selected **EfficientNetB3 CBAM Mixup+CutMix** as the final model.

---
