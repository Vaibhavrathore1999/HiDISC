# Experiments Overview

This section provides comprehensive details about the experimental validation of HIDISC across multiple benchmark datasets. Our experiments demonstrate the effectiveness of hyperbolic geometry for domain generalization with generalized category discovery.

## Experimental Setup

### Evaluation Protocol

We follow the standard **Leave-One-Domain-Out** protocol for domain generalization:
- Train on multiple source domains
- Evaluate on a held-out target domain
- Report average performance across all possible target domains

### Metrics

Our evaluation encompasses multiple metrics to provide a comprehensive assessment:

#### Classification Metrics
- **Accuracy**: Overall classification accuracy on known classes
- **F1-Score**: Harmonic mean of precision and recall
- **Top-k Accuracy**: Accuracy considering top-k predictions

#### Domain Generalization Metrics
- **Domain Gap**: Performance difference between source and target domains
- **Adaptation Rate**: How quickly the model adapts to target domain
- **Robustness Score**: Stability across different domain shifts

#### Novel Category Discovery Metrics
- **Discovery Rate**: Percentage of novel categories successfully identified
- **Clustering Accuracy**: Quality of novel category clustering
- **Purity Score**: Homogeneity of discovered clusters

## Datasets

### 1. PACS Dataset
- **Domains**: Photo, Art Painting, Cartoon, Sketch
- **Classes**: 7 categories (dog, elephant, giraffe, guitar, horse, house, person)
- **Samples**: ~9,991 images total
- **Challenge**: Artistic style variations

### 2. Office-Home Dataset
- **Domains**: Art, Clipart, Product, Real World
- **Classes**: 65 categories
- **Samples**: ~15,588 images total
- **Challenge**: Real-to-synthetic domain gap

### 3. DomainNet Dataset
- **Domains**: Clipart, Infograph, Painting, Quickdraw, Real, Sketch
- **Classes**: 345 categories
- **Samples**: ~600,000 images total
- **Challenge**: Large-scale multi-domain variation

## Experimental Configuration

### Model Architecture

```yaml
backbone:
  type: ResNet50
  pretrained: ImageNet
  frozen_layers: ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']

hyperbolic:
  manifold: Poincaré
  dimension: 128
  curvature: -1.0
  
classifier:
  hidden_dims: [512, 256]
  dropout: 0.5
  activation: ReLU

domain_discriminator:
  hidden_dims: [256, 128]
  gradient_reversal: true
  lambda: 0.1
```

### Training Details

```yaml
training:
  optimizer: SGD
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 1e-4
  batch_size: 32
  epochs: 100
  
scheduling:
  type: CosineAnnealing
  T_max: 100
  eta_min: 1e-6

augmentation:
  random_crop: [224, 224]
  random_horizontal_flip: 0.5
  color_jitter: [0.4, 0.4, 0.4, 0.1]
  random_rotation: 10
```

### Hyperbolic-Specific Settings

```yaml
hyperbolic_training:
  exp_map_eps: 1e-5
  log_map_eps: 1e-5
  riemannian_optimizer: true
  clip_norm: 1.0
  
loss_weights:
  classification: 1.0
  domain_adversarial: 0.1
  hyperbolic_regularization: 0.05
  novel_discovery: 0.2
```

## Baseline Comparisons

We compare HIDISC against several state-of-the-art methods:

### Domain Generalization Baselines
- **ERM (Empirical Risk Minimization)**: Standard supervised learning
- **DANN (Domain Adversarial Neural Networks)**: Adversarial domain adaptation
- **CORAL**: Correlation alignment
- **MLDG**: Meta-learning for domain generalization
- **Mixup**: Data augmentation with mixup
- **GroupDRO**: Distributionally robust optimization

### Category Discovery Baselines
- **K-Means**: Standard clustering
- **SCAN**: Semantic clustering by adopting nearest neighbors
- **SwAV**: Swapping assignments between views
- **GCD**: Generalized category discovery

### Hyperbolic Baselines
- **Hyperbolic NN**: Basic hyperbolic neural networks
- **HGCN**: Hyperbolic graph convolutional networks
- **Poincaré Embeddings**: Hyperbolic embeddings for hierarchical data

## Implementation Details

### Hardware Setup
- **GPUs**: 4x NVIDIA A100 40GB
- **CPU**: Intel Xeon Gold 6248R
- **Memory**: 256GB RAM
- **Storage**: 2TB NVMe SSD

### Software Environment
- **Framework**: PyTorch 1.12.0
- **CUDA**: 11.6
- **Python**: 3.9
- **Hyperbolic Library**: Geoopt 0.4.1

### Reproducibility
- **Random Seeds**: Fixed across all experiments
- **Deterministic Operations**: Enabled for reproducible results
- **Version Control**: All code and configurations tracked

## Hyperparameter Sensitivity

We conducted extensive hyperparameter sensitivity analysis:

### Learning Rate
- **Range**: [1e-5, 1e-1]
- **Optimal**: 1e-3 for most datasets
- **Observation**: Hyperbolic components require careful tuning

### Hyperbolic Dimension
- **Range**: [32, 512]
- **Optimal**: 128-256 depending on dataset complexity
- **Trade-off**: Higher dimensions improve capacity but increase computation

### Curvature
- **Range**: [-2.0, -0.1]
- **Optimal**: -1.0 for most cases
- **Impact**: Significantly affects hyperbolic geometry properties

## Statistical Significance

All results are reported with:
- **Multiple Runs**: 5 independent runs with different random seeds
- **Confidence Intervals**: 95% confidence intervals
- **Statistical Tests**: Paired t-tests for significance testing
- **Effect Size**: Cohen's d for practical significance

## Computational Analysis

### Training Time
- **PACS**: ~2 hours per target domain
- **Office-Home**: ~8 hours per target domain  
- **DomainNet**: ~24 hours per target domain

### Memory Usage
- **Peak GPU Memory**: 8-12GB depending on batch size
- **Model Parameters**: ~25M parameters
- **Storage Requirements**: 100MB model checkpoint

### Scalability
- **Batch Size**: Tested up to 128 with gradient accumulation
- **Multi-GPU**: Linear scaling up to 4 GPUs
- **Dataset Size**: Tested on datasets up to 1M samples

## Visualization and Analysis

### Hyperbolic Embeddings
- **t-SNE**: 2D visualization of hyperbolic embeddings
- **Poincaré Disk**: Native hyperbolic space visualization
- **Domain Clustering**: Visual analysis of domain relationships

### Learning Dynamics
- **Loss Curves**: Training and validation loss progression
- **Accuracy Evolution**: Performance improvement over epochs
- **Discovery Timeline**: Novel category discovery progression

---

**Detailed Results**: Explore specific experimental results for each dataset:
- [PACS Experiments](pacs.md)
- [Office-Home Experiments](office_home.md)
- [DomainNet Experiments](domainnet.md)