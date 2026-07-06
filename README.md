# EEEM071 Coursework — Vehicle Re-Identification (VeRi)


Forked from:  
https://github.com/Surrey-EEEM071-CVDL/Surrey_EEEM071_Coursework

This repository contains the implementation and experimental work completed for the **EEEM071 Advanced Topics in Computer Vision and Deep Learning** coursework. The project focuses on **sequential hyperparameter tuning** of a deep learning pipeline for **vehicle re-identification (VeRi)**, including:

- **Backbone architecture selection**
- **Data augmentation analysis**
- **Hyperparameter optimisation**
- **Performance evaluation using mAP and Rank-1 accuracy**

---

## Repository Structure

```text
├── docs/                              # Documentation and assignment brief from base repo
├── logs/                              # Training logs for each experiment
├── src/                               # Models, datasets, losses, and utilities
├── EEEM071_6954049_Assignment.docx    # Full coursework report
├── args.py                            # Command-line argument definitions
├── main.py                            # Training/evaluation entry point
├── train.sh                           # Example training script
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

---

## Section 1 — Backbone Comparison

### Training Configuration

All backbone architectures were trained using:

- **VeRi dataset**
- **Default data augmentation**
- **Learning rate:** `0.0003`
- **Batch size:** `64`
- **Optimiser:** `AMSGrad`

### Results

| Architecture | mAP (%) | Rank-1 (%) | Notes |
|-------------|----------|------------|-------|
| **mobilenet_v3_small** | 44.1 | 80.2 | Baseline lightweight model |
| **resnet50** | 51.8 | 82.7 | Deeper CNN backbone with residual connections |
| **resnet50_fc512** | 54.1 | 86.7 | Best overall; residual learning with 512-dimensional embedding bottleneck |
| **resnet18_fc512** | 53.8 | 86.2 | Smaller residual network with competitive performance |
| **VGG16** | 23.5 | 61.0 | Underperforms; slower convergence and higher variance |
| **vit_b16-veri** | 22.4 | 51.1 | Vision Transformer struggles on VeRi without stronger inductive bias |

### Key Observation

The **ResNet-50_fc512** architecture achieved the strongest overall performance, providing the best balance between feature representation capability and retrieval accuracy.

---

## Section 2 — Data Augmentation Analysis

### Experimental Setup

The backbone was fixed to:

```text
resnet50_fc512
```

Baseline augmentation consisted of:

- Horizontal Flip
- Random2DTranslation

### Baseline Performance

| Metric | Value |
|---------|--------|
| mAP | 54.1% |
| Rank-1 | 86.7% |

### Augmentation Comparison

| Augmentation Configuration | mAP (%) |
|---------------------------|----------|
| Default (Baseline) | 54.1 |
| + Random Erasing | 53.8 |
| + Colour Jitter | 53.5 |
| + Colour Augmentation | 49.3 |
| + Colour Augmentation + Colour Jitter | 54.3 |

### Key Observation

The combination of:

```text
Colour Augmentation + Colour Jitter
```

produced the highest mAP performance, although the improvement over the baseline was relatively small.

---

## Section 3 — Hyperparameter Tuning

### Experimental Setup

Fixed configuration:

```text
Backbone: ResNet-50_fc512
Augmentation: Colour Augmentation + Colour Jitter
```

### Learning Rate Sweep

| Learning Rate | mAP (%) | Rank-1 (%) | Notes |
|--------------|----------|------------|-------|
| 0.00002 | Slightly lower | Slightly lower | Learning too slowly |
| 0.0001 | 66.0 | 91.2 | Best overall |
| 0.0003 | Lower | Lower | Base repository default |
| 0.001 | Decreased significantly | — | Performance degradation |
| 0.02 | — | — | Training diverged |

#### Observation

A learning rate of `0.0001` provided the best convergence behaviour and highest retrieval accuracy.

### Batch Size Sweep

**Learning Rate Fixed:** `0.0001`

| Batch Size | mAP (%) | Notes |
|-----------|----------|-------|
| 8 | ~57 | Noisy gradient updates |
| 16 | ~57 | Similar behaviour to batch size 8 |
| 64 | 66.0 | Best overall trade-off |
| 128 | 63.4 | Slight performance reduction |

#### Observation

A batch size of `64` achieved the strongest performance and training stability.

### Optimiser Comparison

**Learning Rate:** `0.0001`  
**Batch Size:** `64`

| Optimiser | mAP (%) | Rank-1 (%) | Notes |
|-----------|----------|------------|-------|
| AMSGrad | 66.0 | 91.2 | Best convergence and accuracy |
| SGD | 28.0 | 56.4 | Significant underfitting |

#### Observation

**AMSGrad** substantially outperformed **SGD** under the tested settings, demonstrating superior convergence characteristics for this task.

---

## Best Overall Configuration

### Final Model Setup

| Component | Configuration |
|-----------|--------------|
| Backbone | ResNet-50_fc512 |
| Augmentation | Default + Colour Augmentation + Colour Jitter |
| Learning Rate | 0.0001 |
| Batch Size | 64 |
| Optimiser | AMSGrad |

### Performance

| Metric | Value |
|---------|--------|
| mAP | 66.0% |
| Rank-1 | 91.2% |

This configuration delivered the highest performance achieved during the coursework experiments.

---

## Running the Code

### Reproducing the Best Configuration

```bash
python main.py \
-s veri -t veri \
-a resnet50_fc512 \
--root /path/to/VeRi \
--height 224 --width 224 \
--optim amsgrad --lr 0.0001 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet50_fc512-veri
```

### Reproducing Other Experiments

Modify the following parameters as required:

| Parameter | Purpose |
|-----------|---------|
| `-a` | Backbone architecture |
| `--lr` | Learning rate |
| `--train-batch-size` | Batch size |
| `--optim` | Optimiser |
| Augmentation flags | Data augmentation experiments |

## Report and References

For detailed discussion of:

- Experimental methodology
- Training curve analysis
- Hyperparameter selection rationale
- Vehicle re-identification literature
- MobileNetV3, ResNet, and Vision Transformer architectures
- Data augmentation techniques
- Large-batch training considerations
- AMSGrad and Adam convergence theory

see:

```text
EEEM071_6954049_Assignment.docx
```

which contains the complete written report and reference list.
