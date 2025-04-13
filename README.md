# Adopt

**Overview**:  

The **ADOPT (Adaptive Optimization with Trust)** optimizer is a novel variant of Adam designed to achieve optimal convergence rates with any value of \(\beta2\). It introduces enhancements such as adaptive gradient scaling and cautious updates, making it suitable for diverse optimization scenarios, including tasks requiring stability and robustness in gradient updates.  

This TensorFlow implementation is adapted from the PyTorch version available in the [timm library](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adopt.py). The optimizer builds on concepts from Adam while adding innovative features for enhanced convergence and generalization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.  
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.  
- **`beta2`** *(float, default=0.9999)*: Exponential decay rate for the second moment estimates.  
- **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.  
- **`weight_decay`** *(float, default=0.0)*: Weight decay factor for L2 regularization.  
- **`clip_exp`** *(float, default=0.333)*: Exponent for gradient clipping.  
- **`decoupled`** *(bool, default=False)*: Whether to decouple weight decay from gradient updates.  
- **`caution`** *(bool, default=False)*: Enables cautious updates to prevent overshooting during optimization.  
- **`foreach`** *(bool, default=False)*: If `True`, processes variables in parallel for efficiency.  
- **`maximize`** *(bool, default=False)*: Maximizes the objective function instead of minimizing.  
- **`capturable`** *(bool, default=False)*: Enables capturable state for graph execution.  
- **`differentiable`** *(bool, default=False)*: Ensures the optimizer remains differentiable for higher-order optimization tasks.  
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="adopt")*: Name of the optimizer.

---  

**Example Usage**:  

```python
import tensorflow as tf
from adopt import Adopt

# Initialize the ADOPT optimizer
optimizer = Adopt(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.9999,
    epsilon=1e-6,
    weight_decay=0.01,
    clip_exp=0.333,
    decoupled=True,
    caution=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
