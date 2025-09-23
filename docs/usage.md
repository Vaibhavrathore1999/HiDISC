# Usage Guide

This comprehensive guide demonstrates how to use HIDISC for domain generalization with generalized category discovery.

## Quick Start

### Basic Usage

```python
from hidisc import HIDISCModel, HyperbolicOptimizer
from hidisc.data import DomainDataLoader
from hidisc.training import DomainGeneralizationTrainer

# Initialize the HIDISC model
model = HIDISCModel(
    backbone='resnet50',
    hyperbolic_dim=128,
    num_domains=4,
    num_known_classes=7,
    num_novel_classes=3
)

# Set up hyperbolic optimizer
optimizer = HyperbolicOptimizer(
    model.parameters(),
    lr=0.001,
    manifold='poincare'
)

# Load your data
data_loader = DomainDataLoader(
    dataset='PACS',
    batch_size=32,
    domains=['photo', 'art_painting', 'cartoon', 'sketch']
)

# Train the model
trainer = DomainGeneralizationTrainer(model, optimizer, data_loader)
trainer.train(epochs=100)
```

## Core Components

### 1. Hyperbolic Embeddings

HIDISC leverages hyperbolic geometry for learning hierarchical representations:

```python
from hidisc.geometry import HyperbolicEmbedding

# Create hyperbolic embedding layer
h_embedding = HyperbolicEmbedding(
    input_dim=2048,  # ResNet50 features
    hyperbolic_dim=128,
    curvature=-1.0
)

# Project features to hyperbolic space
features = backbone(images)  # Shape: [batch_size, 2048]
h_features = h_embedding(features)  # Shape: [batch_size, 128]
```

### 2. Domain-Aware Learning

Configure domain-specific components:

```python
from hidisc.domain import DomainClassifier, DomainAligner

# Domain classification for adversarial training
domain_classifier = DomainClassifier(
    input_dim=128,
    num_domains=4,
    hidden_dims=[64, 32]
)

# Domain alignment in hyperbolic space
domain_aligner = DomainAligner(
    manifold='poincare',
    alignment_strength=0.1
)
```

### 3. Category Discovery

Enable discovery of novel categories:

```python
from hidisc.discovery import NovelCategoryDiscovery

# Configure category discovery
category_discovery = NovelCategoryDiscovery(
    num_known_classes=7,
    max_novel_classes=5,
    discovery_threshold=0.8,
    clustering_method='hyperbolic_kmeans'
)

# Discover novel categories
novel_predictions = category_discovery.discover(h_features, predictions)
```

## Training Procedures

### Standard Training

```python
# Basic training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        images, labels, domains = batch
        
        # Forward pass
        outputs = model(images, domains)
        
        # Compute losses
        classification_loss = criterion(outputs.logits, labels)
        domain_loss = domain_criterion(outputs.domain_logits, domains)
        hyperbolic_loss = hyperbolic_criterion(outputs.h_features)
        
        # Total loss
        total_loss = classification_loss + domain_loss + hyperbolic_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

### Advanced Training with Category Discovery

```python
from hidisc.training import HIDISCTrainer

trainer = HIDISCTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    config={
        'classification_weight': 1.0,
        'domain_weight': 0.1,
        'hyperbolic_weight': 0.05,
        'discovery_weight': 0.2,
        'discovery_start_epoch': 20
    }
)

# Train with automatic category discovery
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_checkpoint_every=10
)
```

## Configuration

### Model Configuration

```yaml
# config.yaml
model:
  backbone: 'resnet50'
  pretrained: true
  hyperbolic_dim: 128
  curvature: -1.0
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  weight_decay: 1e-4
  
hyperbolic:
  manifold: 'poincare'
  exp_map_eps: 1e-5
  log_map_eps: 1e-5
  
discovery:
  enable: true
  start_epoch: 20
  threshold: 0.8
  max_novel_classes: 5
```

### Loading Configuration

```python
from hidisc.config import load_config

config = load_config('config.yaml')
model = HIDISCModel.from_config(config)
```

## Evaluation

### Standard Evaluation

```python
from hidisc.evaluation import DomainGeneralizationEvaluator

evaluator = DomainGeneralizationEvaluator(model)

# Evaluate on test domains
results = evaluator.evaluate(
    test_loader=test_loader,
    target_domains=['sketch'],  # Unseen domain
    metrics=['accuracy', 'f1_score', 'novel_discovery_rate']
)

print(f"Target domain accuracy: {results['accuracy']:.3f}")
print(f"Novel category discovery rate: {results['novel_discovery_rate']:.3f}")
```

### Comprehensive Analysis

```python
from hidisc.analysis import HyperbolicAnalyzer

analyzer = HyperbolicAnalyzer(model)

# Visualize hyperbolic embeddings
analyzer.plot_hyperbolic_embeddings(
    features=test_features,
    labels=test_labels,
    domains=test_domains,
    save_path='embeddings.png'
)

# Analyze domain alignment
alignment_scores = analyzer.compute_domain_alignment(
    source_features=source_features,
    target_features=target_features
)
```

## Best Practices

### 1. Data Preparation

```python
# Ensure proper data normalization
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 2. Hyperparameter Tuning

```python
# Use learning rate scheduling
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

### 3. Memory Optimization

```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images, domains)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Examples and Tutorials

### Example 1: PACS Dataset

```python
from hidisc.examples import pacs_example

# Run PACS experiment
results = pacs_example.run(
    target_domain='sketch',
    num_epochs=100,
    save_results=True
)
```

### Example 2: Custom Dataset

```python
from hidisc.data import CustomDataset

# Define your custom dataset
class MyDataset(CustomDataset):
    def __init__(self, data_path, domains):
        super().__init__(data_path, domains)
        # Custom initialization
    
    def get_domain_info(self, idx):
        # Return domain information for sample idx
        pass

# Use with HIDISC
dataset = MyDataset('path/to/data', domains=['domain1', 'domain2'])
```

## Command Line Interface

HIDISC also provides a convenient CLI:

```bash
# Train on PACS dataset
hidisc train --dataset PACS --target-domain sketch --epochs 100

# Evaluate trained model
hidisc evaluate --model-path checkpoints/best_model.pth --dataset PACS

# Run hyperparameter search
hidisc tune --config tune_config.yaml --trials 50
```

## Troubleshooting

### Common Issues

1. **Numerical Instability in Hyperbolic Operations**
   ```python
   # Use higher precision
   torch.set_default_dtype(torch.float64)
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Use gradient accumulation
   accumulate_grad_batches = 4
   ```

3. **Slow Convergence**
   ```python
   # Adjust learning rates for different components
   optimizer = HyperbolicOptimizer([
       {'params': model.backbone.parameters(), 'lr': 0.0001},
       {'params': model.hyperbolic_layers.parameters(), 'lr': 0.001}
   ])
   ```

---

**Next Steps**: Explore our [Experiments](experiments/index.md) section to see detailed experimental setups and results.