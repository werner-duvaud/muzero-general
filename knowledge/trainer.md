# Trainer Module

## Overview

The **Trainer** module is the core training engine of the MuZero system, responsible for continuously optimizing the neural network using experience data collected from self-play games. Running as a dedicated Ray actor in a separate thread, the Trainer implements the MuZero training algorithm with support for prioritized experience replay (PER), learning rate scheduling, and multi-step unrolling for temporal credit assignment.

The module bridges the gap between collected game experiences (stored in the [replay buffer](replay_buffer.md)) and the neural network predictions (from [neural networks](neural_networks.md)), implementing the sophisticated loss functions and optimization strategies that enable MuZero to learn effective value, policy, and dynamics models simultaneously.

### Key Responsibilities

- **Continuous Training Loop**: Maintains an ongoing training process that runs concurrently with self-play
- **Batch Processing**: Retrieves and processes batches of game experiences from the replay buffer
- **Multi-Head Loss Computation**: Calculates and optimizes value, reward, and policy losses simultaneously
- **Model Unrolling**: Performs recurrent inference through the dynamics model for multi-step predictions
- **Priority Updates**: Computes and updates priorities for prioritized experience replay
- **Learning Rate Scheduling**: Implements exponential decay of learning rate over training steps
- **Checkpoint Management**: Periodically saves model weights and optimizer state to shared storage
- **Training Ratio Control**: Manages the balance between self-play and training speeds

---

## Architecture

### System Context

```mermaid
graph TB
    subgraph "MuZero System"
        SP[Self-Play Workers<br/>self_play_mcts.md]
        RB[Replay Buffer<br/>replay_buffer.md]
        TR[Trainer<br/>Current Module]
        SS[Shared Storage<br/>shared_storage.md]
        NN[Neural Networks<br/>neural_networks.md]
    end
    
    SP -->|Save Game Histories| RB
    RB -->|Sample Batches| TR
    TR -->|Update Priorities| RB
    TR -->|Save Weights & Metrics| SS
    SS -->|Load Initial Checkpoint| TR
    TR -->|Train & Optimize| NN
    NN -->|Forward/Backward Pass| TR
    SS -->|Provide Weights| SP
    
    style TR fill:#e1f5ff,stroke:#0066cc,stroke-width:3px
    style RB fill:#f0f0f0,stroke:#666,stroke-width:2px
    style SS fill:#f0f0f0,stroke:#666,stroke-width:2px
    style NN fill:#f0f0f0,stroke:#666,stroke-width:2px
    style SP fill:#f0f0f0,stroke:#666,stroke-width:2px
```

### Component Architecture

```mermaid
classDiagram
    class Trainer {
        -config: MuZeroConfig
        -model: MuZeroNetwork
        -optimizer: Optimizer
        -training_step: int
        +__init__(initial_checkpoint, config)
        +continuous_update_weights(replay_buffer, shared_storage)
        +update_weights(batch) tuple
        +update_lr() void
        +loss_function(value, reward, policy_logits, targets) tuple
    }
    
    class MuZeroNetwork {
        +initial_inference(observation)
        +recurrent_inference(hidden_state, action)
        +get_weights()
        +set_weights(weights)
    }
    
    class ReplayBuffer {
        +get_batch() tuple
        +update_priorities(priorities, indices)
    }
    
    class SharedStorage {
        +get_info(key)
        +set_info(key, value)
        +save_checkpoint()
    }
    
    class Optimizer {
        +zero_grad()
        +step()
        +load_state_dict(state)
        +state_dict()
    }
    
    Trainer --> MuZeroNetwork : trains
    Trainer --> ReplayBuffer : samples from
    Trainer --> SharedStorage : saves to
    Trainer --> Optimizer : uses
    
    note for Trainer "Ray Remote Actor\nRuns in dedicated thread"
```

---

## Core Components

### Trainer Class

The `Trainer` class is implemented as a Ray remote actor (`@ray.remote`) that runs in a dedicated thread, enabling parallel execution with self-play workers.

#### Initialization

```python
def __init__(self, initial_checkpoint, config):
    """
    Initialize the trainer with a checkpoint and configuration.
    
    Args:
        initial_checkpoint: Dict containing:
            - weights: Initial neural network weights
            - optimizer_state: Previous optimizer state (or None)
            - training_step: Current training step count
        config: MuZeroConfig object with training hyperparameters
    """
```

**Initialization Process:**

1. **Random Seed Setup**: Fixes NumPy and PyTorch random seeds for reproducibility
2. **Network Initialization**: Creates MuZeroNetwork and loads initial weights
3. **Device Configuration**: Moves model to GPU if configured and available
4. **Training Mode**: Sets model to training mode (enables dropout, batch norm training)
5. **Optimizer Setup**: Initializes SGD or Adam optimizer with configured hyperparameters
6. **State Restoration**: Loads previous optimizer state if resuming training

**Supported Optimizers:**
- **SGD**: Stochastic Gradient Descent with momentum and weight decay
- **Adam**: Adaptive moment estimation with weight decay

---

## Training Process

### Continuous Training Loop

```mermaid
flowchart TD
    Start([Start Training]) --> Wait{Replay Buffer<br/>Has Games?}
    Wait -->|No| Sleep1[Sleep 0.1s]
    Sleep1 --> Wait
    Wait -->|Yes| RequestBatch[Request Next Batch<br/>from Replay Buffer]
    
    RequestBatch --> Loop{Training Step <<br/>Max Steps?}
    Loop -->|No| GetBatch[Get Current Batch]
    GetBatch --> RequestNext[Request Next Batch<br/>Async]
    RequestNext --> UpdateLR[Update Learning Rate]
    UpdateLR --> Train[Update Weights<br/>Forward + Backward Pass]
    
    Train --> PER{PER Enabled?}
    PER -->|Yes| UpdatePriorities[Update Priorities<br/>in Replay Buffer]
    PER -->|No| CheckSave
    UpdatePriorities --> CheckSave{Checkpoint<br/>Interval?}
    
    CheckSave -->|Yes| SaveWeights[Save Weights &<br/>Optimizer State]
    SaveWeights --> CheckModel{Save Model<br/>Enabled?}
    CheckModel -->|Yes| SaveCheckpoint[Save Checkpoint<br/>to Disk]
    CheckModel -->|No| SaveMetrics
    CheckSave -->|No| SaveMetrics[Save Training Metrics<br/>to Shared Storage]
    SaveCheckpoint --> SaveMetrics
    
    SaveMetrics --> Delay{Training<br/>Delay?}
    Delay -->|Yes| Sleep2[Sleep Configured Time]
    Delay -->|No| CheckRatio
    Sleep2 --> CheckRatio{Ratio<br/>Control?}
    
    CheckRatio -->|Yes| RatioCheck{Training/Self-Play<br/>Ratio > Target?}
    RatioCheck -->|Yes| Sleep3[Sleep 0.5s]
    Sleep3 --> RatioCheck
    RatioCheck -->|No| Loop
    CheckRatio -->|No| Loop
    
    Loop -->|Yes| End([Training Complete])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Train fill:#FFE4B5
    style UpdatePriorities fill:#E6E6FA
    style SaveWeights fill:#FFE4E1
```

### Weight Update Process

The `update_weights` method performs a single training step with the following stages:

```mermaid
flowchart TD
    Start([Receive Batch]) --> Unpack[Unpack Batch Data:<br/>observations, actions,<br/>targets, weights]
    Unpack --> Convert[Convert to Tensors<br/>& Move to Device]
    Convert --> Transform[Transform Targets<br/>to Support Representation]
    
    Transform --> Initial[Initial Inference:<br/>representation → prediction]
    Initial --> Store1[Store Prediction 0]
    Store1 --> Unroll{More Unroll<br/>Steps?}
    
    Unroll -->|Yes| Recurrent[Recurrent Inference:<br/>hidden_state + action → prediction]
    Recurrent --> Hook[Register Gradient Hook<br/>Scale by 0.5]
    Hook --> Store2[Store Prediction i]
    Store2 --> Unroll
    
    Unroll -->|No| ComputeLoss[Compute Losses for<br/>All Predictions]
    ComputeLoss --> Loss0[Step 0: Value + Policy Loss<br/>No Reward Loss]
    Loss0 --> LossN[Steps 1-N: Value + Reward + Policy<br/>with Gradient Scaling]
    
    LossN --> Scale[Scale Value Loss<br/>by Weight Factor]
    Scale --> PER{PER Enabled?}
    PER -->|Yes| Weight[Apply Importance<br/>Sampling Weights]
    PER -->|No| Mean
    Weight --> Mean[Mean Over Batch]
    
    Mean --> Backward[Backward Pass:<br/>Compute Gradients]
    Backward --> Optimize[Optimizer Step:<br/>Update Weights]
    Optimize --> Increment[Increment Training Step]
    Increment --> Return[Return Priorities<br/>& Loss Values]
    Return --> End([Complete])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Initial fill:#FFE4B5
    style Recurrent fill:#FFE4B5
    style ComputeLoss fill:#E6E6FA
    style Backward fill:#FFD700
    style Optimize fill:#FFD700
```

#### Detailed Steps

**1. Data Preparation**
```python
# Unpack batch components
observation_batch, action_batch, target_value, target_reward, 
target_policy, weight_batch, gradient_scale_batch = batch

# Convert to tensors and move to device
observation_batch = torch.tensor(observation_batch).float().to(device)
action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
# ... (other conversions)

# Transform scalar targets to support representation
target_value = models.scalar_to_support(target_value, support_size)
target_reward = models.scalar_to_support(target_reward, support_size)
```

**2. Generate Predictions via Unrolling**
```python
# Initial inference from observation
value, reward, policy_logits, hidden_state = model.initial_inference(observation_batch)
predictions = [(value, reward, policy_logits)]

# Recurrent inference for K unroll steps
for i in range(1, num_unroll_steps + 1):
    value, reward, policy_logits, hidden_state = model.recurrent_inference(
        hidden_state, action_batch[:, i]
    )
    # Scale dynamics gradient (paper appendix)
    hidden_state.register_hook(lambda grad: grad * 0.5)
    predictions.append((value, reward, policy_logits))
```

**3. Compute Multi-Step Losses**

The loss function uses cross-entropy for all three heads:

```python
# Cross-entropy loss (better convergence than MSE)
value_loss = (-target_value * LogSoftmax(value)).sum(1)
reward_loss = (-target_reward * LogSoftmax(reward)).sum(1)
policy_loss = (-target_policy * LogSoftmax(policy_logits)).sum(1)
```

**Special handling for first step:**
- Reward loss is ignored (no previous action to predict reward from)
- Only value and policy losses are computed

**Gradient scaling for unroll steps:**
```python
# Scale by number of valid unroll steps
current_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
```

**4. Priority Computation (PER)**

For each prediction step, compute TD-error based priorities:
```python
pred_value_scalar = support_to_scalar(value).detach().cpu().numpy()
priorities[:, i] = abs(pred_value_scalar - target_value_scalar[:, i]) ** PER_alpha
```

**5. Final Loss Aggregation**
```python
# Combine losses with value loss scaling
loss = value_loss * value_loss_weight + reward_loss + policy_loss

# Apply importance sampling weights (PER)
if PER:
    loss *= weight_batch

# Mean over batch
loss = loss.mean()
```

**6. Optimization**
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
training_step += 1
```

---

## Loss Functions

### Multi-Head Loss Architecture

```mermaid
graph TB
    subgraph "Predictions"
        V[Value Distribution<br/>2*support_size+1]
        R[Reward Distribution<br/>2*support_size+1]
        P[Policy Logits<br/>action_space_size]
    end
    
    subgraph "Targets"
        TV[Target Value<br/>Support Distribution]
        TR[Target Reward<br/>Support Distribution]
        TP[Target Policy<br/>Visit Counts]
    end
    
    subgraph "Loss Computation"
        VL[Value Loss<br/>Cross-Entropy]
        RL[Reward Loss<br/>Cross-Entropy]
        PL[Policy Loss<br/>Cross-Entropy]
    end
    
    V --> VL
    TV --> VL
    R --> RL
    TR --> RL
    P --> PL
    TP --> PL
    
    VL --> Scale[Scale Value Loss<br/>× value_loss_weight]
    RL --> Combine[Combined Loss]
    PL --> Combine
    Scale --> Combine
    
    Combine --> PER{PER?}
    PER -->|Yes| ISW[× IS Weights]
    PER -->|No| Mean[Mean]
    ISW --> Mean
    Mean --> Final[Final Loss]
    
    style V fill:#FFE4B5
    style R fill:#FFE4B5
    style P fill:#FFE4B5
    style VL fill:#FFB6C1
    style RL fill:#FFB6C1
    style PL fill:#FFB6C1
    style Final fill:#90EE90
```

### Loss Function Details

#### Value Loss
- **Purpose**: Train the value network to predict expected returns
- **Type**: Cross-entropy between predicted and target value distributions
- **Scaling**: Multiplied by `value_loss_weight` (typically 0.25 per paper)
- **Target**: Discounted sum of future rewards + bootstrap value from MCTS

#### Reward Loss
- **Purpose**: Train the dynamics model to predict immediate rewards
- **Type**: Cross-entropy between predicted and target reward distributions
- **Special Case**: Ignored for the first step (no previous action)
- **Target**: Actual reward received after taking action

#### Policy Loss
- **Purpose**: Train the policy network to match MCTS visit distributions
- **Type**: Cross-entropy between predicted policy and MCTS visit counts
- **Target**: Normalized visit counts from MCTS search

#### Support Representation

MuZero uses a categorical representation for value and reward:

```mermaid
graph LR
    Scalar[Scalar Value<br/>e.g., 3.7] --> Transform[Transform Function<br/>h of x with scaling]
    Transform --> Discretize[Discretize to Support<br/>range: -support_size to +support_size]
    Discretize --> Distribution[Probability Distribution<br/>2*support_size+1 bins]
    
    style Scalar fill:#FFE4B5
    style Distribution fill:#90EE90
```

**Benefits:**
- More stable training than regression
- Better representation of value uncertainty
- Improved gradient flow

---

## Learning Rate Scheduling

### Exponential Decay Strategy

```mermaid
graph LR
    Start[Initial LR<br/>lr_init] --> Decay[Exponential Decay<br/>lr = lr_init × decay_rate^step/decay_steps]
    Decay --> Update[Update All<br/>Parameter Groups]
    Update --> Apply[Apply to<br/>Optimizer]
    
    style Start fill:#FFE4B5
    style Decay fill:#E6E6FA
    style Apply fill:#90EE90
```

**Formula:**
```python
lr = lr_init * lr_decay_rate ** (training_step / lr_decay_steps)
```

**Typical Configuration:**
- `lr_init`: 0.001 - 0.02
- `lr_decay_rate`: 0.1 - 0.9
- `lr_decay_steps`: 10,000 - 350,000

**Update Frequency:** Every training step before weight update

---

## Gradient Management

### Gradient Scaling Strategies

```mermaid
graph TB
    subgraph "Dynamics Gradient Scaling"
        HS[Hidden State] --> Hook1[Register Hook<br/>× 0.5]
        Hook1 --> Scaled1[Scaled Gradient]
    end
    
    subgraph "Unroll Step Scaling"
        Loss[Step Loss] --> Hook2[Register Hook<br/>÷ gradient_scale]
        Hook2 --> Scaled2[Scaled Gradient]
    end
    
    subgraph "Value Loss Scaling"
        VL[Value Loss] --> Multiply[× value_loss_weight]
        Multiply --> Scaled3[Scaled Loss]
    end
    
    Scaled1 --> Backward[Backward Pass]
    Scaled2 --> Backward
    Scaled3 --> Backward
    Backward --> Gradients[Final Gradients]
    
    style Hook1 fill:#FFE4B5
    style Hook2 fill:#FFE4B5
    style Multiply fill:#FFE4B5
    style Gradients fill:#90EE90
```

#### 1. Dynamics Function Gradient Scaling

**Purpose:** Prevent the dynamics function from dominating the gradient flow

**Implementation:**
```python
hidden_state.register_hook(lambda grad: grad * 0.5)
```

**Rationale:** The dynamics function is applied recursively, so its gradients can become disproportionately large

#### 2. Unroll Step Gradient Scaling

**Purpose:** Normalize gradients based on the number of valid unroll steps

**Implementation:**
```python
loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
```

**Gradient Scale Calculation:**
```python
gradient_scale = min(num_unroll_steps, len(game_history) - current_position)
```

**Rationale:** Games of different lengths contribute different numbers of unroll steps; this normalizes their contribution

#### 3. Value Loss Weighting

**Purpose:** Balance the relative importance of value, reward, and policy losses

**Implementation:**
```python
loss = value_loss * value_loss_weight + reward_loss + policy_loss
```

**Typical Value:** 0.25 (as recommended in the MuZero paper appendix)

---

## Prioritized Experience Replay (PER)

### Priority Computation and Update Flow

```mermaid
sequenceDiagram
    participant T as Trainer
    participant RB as Replay Buffer
    participant GH as Game History
    
    Note over RB: Sample batch with<br/>game priorities
    RB->>T: Batch + IS weights
    
    Note over T: Forward pass<br/>Generate predictions
    T->>T: Compute TD errors<br/>for each step
    
    Note over T: Calculate priorities<br/>|TD-error|^α
    T->>RB: Update priorities<br/>(game_id, position, priorities)
    
    RB->>GH: Update position priorities
    RB->>GH: Update game priority<br/>(max of positions)
    
    Note over RB: Updated priorities used<br/>in next sampling
```

### Priority Calculation

**TD-Error Based Priorities:**
```python
# For each unroll step
pred_value = support_to_scalar(value_prediction)
target_value = target_value_scalar[:, i]
td_error = abs(pred_value - target_value)
priority = td_error ** PER_alpha
```

**Parameters:**
- `PER_alpha`: Controls priority exponent (typically 0.6 - 1.0)
- Higher alpha = more aggressive prioritization

### Importance Sampling Correction

**Purpose:** Correct bias introduced by non-uniform sampling

**Weight Calculation:**
```python
# In replay buffer during sampling
weight = 1 / (total_samples * game_prob * position_prob)

# Normalize by max weight
weight_batch = weight_batch / max(weight_batch)
```

**Application:**
```python
# In loss computation
loss *= weight_batch
```

---

## Training Ratio Control

### Self-Play vs Training Balance

```mermaid
graph TB
    Start[Training Step Complete] --> Delay{Training<br/>Delay Set?}
    Delay -->|Yes| Sleep1[Sleep for<br/>training_delay seconds]
    Delay -->|No| CheckRatio
    Sleep1 --> CheckRatio{Ratio<br/>Control Enabled?}
    
    CheckRatio -->|No| Continue[Continue Training]
    CheckRatio -->|Yes| CalcRatio[Calculate Ratio:<br/>training_steps / played_steps]
    
    CalcRatio --> Compare{Ratio ><br/>Target?}
    Compare -->|No| Continue
    Compare -->|Yes| Wait[Wait for<br/>More Self-Play]
    Wait --> Sleep2[Sleep 0.5s]
    Sleep2 --> CheckTerm{Terminate<br/>Signal?}
    CheckTerm -->|Yes| Stop[Stop Training]
    CheckTerm -->|No| Update[Get Latest<br/>played_steps]
    Update --> Compare
    
    Continue --> Next[Next Training Step]
    
    style Start fill:#90EE90
    style Wait fill:#FFE4B5
    style Continue fill:#90EE90
    style Stop fill:#FFB6C1
```

### Configuration Options

**1. Training Delay**
```python
if config.training_delay:
    time.sleep(config.training_delay)
```
- Fixed delay after each training step
- Simple but inflexible approach

**2. Ratio Control**
```python
ratio = training_steps / max(1, num_played_steps)
if ratio > config.ratio:
    # Wait for more self-play data
    time.sleep(0.5)
```
- Dynamic balancing based on actual progress
- Prevents training from outpacing data collection
- Typical ratio: 0.1 - 1.0

**Benefits:**
- Prevents overfitting to limited data
- Ensures fresh experiences are available
- Balances computational resources

---

## Checkpoint Management

### Save Strategy

```mermaid
flowchart TD
    Step[Training Step] --> Check{Step %<br/>checkpoint_interval<br/>== 0?}
    Check -->|No| Continue[Continue Training]
    Check -->|Yes| Prepare[Prepare Checkpoint Data]
    
    Prepare --> Weights[Copy Model Weights<br/>to CPU]
    Weights --> Optimizer[Copy Optimizer State<br/>to CPU]
    Optimizer --> Send[Send to Shared Storage]
    
    Send --> SaveModel{save_model<br/>Enabled?}
    SaveModel -->|Yes| Disk[Save Checkpoint<br/>to Disk]
    SaveModel -->|No| Continue
    Disk --> Continue
    
    style Check fill:#FFE4B5
    style Weights fill:#E6E6FA
    style Optimizer fill:#E6E6FA
    style Disk fill:#90EE90
```

### Checkpoint Contents

**Saved to Shared Storage (Every Interval):**
```python
{
    "weights": model.get_weights(),           # Neural network parameters
    "optimizer_state": optimizer.state_dict() # Optimizer state (momentum, etc.)
}
```

**Saved to Disk (If Enabled):**
- Complete checkpoint including weights and optimizer state
- Allows resuming training from any checkpoint
- Stored at `config.results_path / "model.checkpoint"`

**Checkpoint Interval:**
- Configurable via `config.checkpoint_interval`
- Typical values: 10 - 1000 steps
- Trade-off between overhead and recovery granularity

---

## Training Metrics

### Metrics Tracked and Reported

```mermaid
graph TB
    subgraph "Loss Metrics"
        TL[Total Loss<br/>Combined weighted loss]
        VL[Value Loss<br/>Value prediction error]
        RL[Reward Loss<br/>Reward prediction error]
        PL[Policy Loss<br/>Policy matching error]
    end
    
    subgraph "Training Progress"
        TS[Training Step<br/>Current iteration]
        LR[Learning Rate<br/>Current LR value]
    end
    
    subgraph "Shared Storage"
        SS[SharedStorage]
    end
    
    TL --> SS
    VL --> SS
    RL --> SS
    PL --> SS
    TS --> SS
    LR --> SS
    
    SS --> Logging[Logging System]
    SS --> Monitoring[Monitoring Dashboard]
    
    style TL fill:#FFB6C1
    style VL fill:#FFE4B5
    style RL fill:#FFE4B5
    style PL fill:#FFE4B5
    style SS fill:#90EE90
```

### Metrics Saved Each Step

```python
shared_storage.set_info.remote({
    "training_step": self.training_step,
    "lr": self.optimizer.param_groups[0]["lr"],
    "total_loss": total_loss,
    "value_loss": value_loss,
    "reward_loss": reward_loss,
    "policy_loss": policy_loss,
})
```

**Metric Descriptions:**

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| `training_step` | Current training iteration | 0 - training_steps |
| `lr` | Current learning rate | 0.0001 - 0.02 |
| `total_loss` | Combined weighted loss | 0.1 - 10.0 |
| `value_loss` | Value prediction loss | 0.05 - 5.0 |
| `reward_loss` | Reward prediction loss | 0.01 - 2.0 |
| `policy_loss` | Policy matching loss | 0.05 - 5.0 |

**Usage:**
- Monitoring training progress
- Detecting training instabilities
- Hyperparameter tuning
- Debugging convergence issues

---

## Data Flow

### Complete Training Data Pipeline

```mermaid
flowchart LR
    subgraph "Replay Buffer"
        Games[(Game<br/>Histories)]
        Sample[Sample Batch]
        Games --> Sample
    end
    
    subgraph "Trainer"
        Receive[Receive Batch]
        Prepare[Prepare Tensors]
        Forward[Forward Pass<br/>Initial + Unroll]
        Loss[Compute Losses]
        Backward[Backward Pass]
        Update[Update Weights]
        Priority[Compute Priorities]
        
        Receive --> Prepare
        Prepare --> Forward
        Forward --> Loss
        Loss --> Backward
        Backward --> Update
        Loss --> Priority
    end
    
    subgraph "Shared Storage"
        Weights[Model Weights]
        Metrics[Training Metrics]
    end
    
    Sample -->|Batch + IS Weights| Receive
    Priority -->|New Priorities| Games
    Update -->|Updated Weights| Weights
    Update -->|Loss Values| Metrics
    
    style Forward fill:#FFE4B5
    style Loss fill:#FFB6C1
    style Update fill:#90EE90
```

### Batch Data Structure

**Input Batch:**
```python
(
    observation_batch,      # [batch, channels, height, width]
    action_batch,          # [batch, num_unroll_steps+1]
    target_value,          # [batch, num_unroll_steps+1]
    target_reward,         # [batch, num_unroll_steps+1]
    target_policy,         # [batch, num_unroll_steps+1, action_space]
    weight_batch,          # [batch] (if PER enabled)
    gradient_scale_batch   # [batch, num_unroll_steps+1]
)
```

**Output:**
```python
(
    priorities,            # [batch, num_unroll_steps+1]
    total_loss,           # scalar
    value_loss,           # scalar
    reward_loss,          # scalar
    policy_loss           # scalar
)
```

---

## Configuration Parameters

### Key Training Hyperparameters

```python
class MuZeroConfig:
    # Optimizer settings
    optimizer: str = "Adam"              # "SGD" or "Adam"
    lr_init: float = 0.005              # Initial learning rate
    lr_decay_rate: float = 0.1          # LR decay rate
    lr_decay_steps: int = 350000        # Steps for one decay cycle
    momentum: float = 0.9               # SGD momentum (if using SGD)
    weight_decay: float = 1e-4          # L2 regularization
    
    # Training loop
    training_steps: int = 1000000       # Total training steps
    checkpoint_interval: int = 100      # Steps between checkpoints
    save_model: bool = True             # Save checkpoints to disk
    train_on_gpu: bool = True           # Use GPU for training
    
    # Batch configuration
    batch_size: int = 128               # Samples per training step
    num_unroll_steps: int = 5           # Steps to unroll dynamics
    td_steps: int = 10                  # Steps for TD target
    
    # Loss weights
    value_loss_weight: float = 0.25     # Scale factor for value loss
    
    # PER settings
    PER: bool = True                    # Enable prioritized replay
    PER_alpha: float = 0.6              # Priority exponent
    
    # Training ratio control
    ratio: float = None                 # Training/self-play ratio (None=disabled)
    training_delay: float = 0           # Fixed delay per step (seconds)
    
    # Support size for value/reward
    support_size: int = 300             # Categorical support range
    
    # Random seed
    seed: int = 0                       # Random seed for reproducibility
```

### Parameter Tuning Guidelines

**Learning Rate:**
- Start with `lr_init = 0.005` for most games
- Increase for simple games, decrease for complex ones
- Monitor loss curves for instability

**Batch Size:**
- Larger batches = more stable gradients but slower iteration
- Typical range: 64 - 512
- Limited by GPU memory

**Unroll Steps:**
- More steps = better credit assignment but more computation
- Typical range: 5 - 10
- Should be ≤ typical game length

**Value Loss Weight:**
- Paper recommends 0.25
- Increase if value predictions are poor
- Decrease if value dominates training

**PER Alpha:**
- Higher = more aggressive prioritization
- 0.0 = uniform sampling
- 0.6 - 1.0 typical range

---

## Integration with Other Modules

### Module Dependencies

```mermaid
graph TB
    subgraph "Trainer Dependencies"
        T[Trainer]
    end
    
    subgraph "Required Modules"
        NN[Neural Networks<br/>neural_networks.md]
        RB[Replay Buffer<br/>replay_buffer.md]
        SS[Shared Storage<br/>shared_storage.md]
    end
    
    subgraph "External Libraries"
        Ray[Ray<br/>Distributed Computing]
        Torch[PyTorch<br/>Deep Learning]
        Numpy[NumPy<br/>Numerical Computing]
    end
    
    T -->|Trains| NN
    T -->|Samples From| RB
    T -->|Saves To| SS
    T -->|Uses| Ray
    T -->|Uses| Torch
    T -->|Uses| Numpy
    
    RB -->|Provides Batches| T
    SS -->|Provides Initial Checkpoint| T
    NN -->|Provides Predictions| T
    
    style T fill:#e1f5ff,stroke:#0066cc,stroke-width:3px
```

### Communication Patterns

**With Replay Buffer:**
```python
# Asynchronous batch retrieval
next_batch = replay_buffer.get_batch.remote()
index_batch, batch = ray.get(next_batch)

# Priority updates
replay_buffer.update_priorities.remote(priorities, index_batch)
```

**With Shared Storage:**
```python
# Read operations
num_games = ray.get(shared_storage.get_info.remote("num_played_games"))
terminate = ray.get(shared_storage.get_info.remote("terminate"))

# Write operations
shared_storage.set_info.remote({
    "weights": model.get_weights(),
    "training_step": training_step,
    "total_loss": total_loss
})
shared_storage.save_checkpoint.remote()
```

**With Neural Networks:**
```python
# Model operations
model = models.MuZeroNetwork(config)
model.set_weights(checkpoint["weights"])
model.to(device)
model.train()

# Inference
value, reward, policy, hidden = model.initial_inference(observation)
value, reward, policy, hidden = model.recurrent_inference(hidden, action)
```

---

## Performance Considerations

### Optimization Strategies

```mermaid
graph TB
    subgraph "GPU Utilization"
        GPU1[Train on GPU<br/>train_on_gpu=True]
        GPU2[Batch Size<br/>Maximize GPU memory]
        GPU3[Mixed Precision<br/>Optional]
    end
    
    subgraph "Asynchronous Operations"
        Async1[Prefetch Next Batch<br/>While training current]
        Async2[Async Priority Updates<br/>Non-blocking]
        Async3[Async Checkpoint Saves<br/>Non-blocking]
    end
    
    subgraph "Computational Efficiency"
        Comp1[Checkpoint Interval<br/>Balance overhead]
        Comp2[Gradient Accumulation<br/>Optional]
        Comp3[Unroll Steps<br/>Balance accuracy/speed]
    end
    
    GPU1 --> Perf[Training Performance]
    GPU2 --> Perf
    GPU3 --> Perf
    Async1 --> Perf
    Async2 --> Perf
    Async3 --> Perf
    Comp1 --> Perf
    Comp2 --> Perf
    Comp3 --> Perf
    
    style Perf fill:#90EE90
```

### Performance Tips

**1. GPU Utilization**
- Always train on GPU if available (`train_on_gpu=True`)
- Maximize batch size within GPU memory limits
- Monitor GPU utilization (should be >80%)

**2. Asynchronous Batch Retrieval**
```python
# Prefetch next batch while training current
next_batch = replay_buffer.get_batch.remote()
# ... train on current batch ...
index_batch, batch = ray.get(next_batch)  # Minimal wait
```

**3. Checkpoint Frequency**
- More frequent = more overhead but better recovery
- Less frequent = less overhead but more lost progress
- Typical: every 100-1000 steps

**4. Training Ratio**
- Set `ratio` to prevent training from outpacing self-play
- Typical: 0.1 - 1.0 (training steps per self-play step)
- Monitor `training_step / num_played_steps`

**5. Batch Size Scaling**
- Larger batches improve GPU utilization
- May need to adjust learning rate (linear scaling rule)
- Limited by GPU memory

### Typical Performance Metrics

| Configuration | Training Speed | GPU Utilization |
|---------------|----------------|-----------------|
| Small Network (FC) | 100-200 steps/sec | 60-80% |
| Medium Network (ResNet) | 20-50 steps/sec | 80-95% |
| Large Network (ResNet) | 5-15 steps/sec | 90-99% |

*Note: Speeds depend on hardware, batch size, and unroll steps*

---

## Error Handling and Edge Cases

### Common Issues and Solutions

```mermaid
graph TB
    subgraph "Training Issues"
        Issue1[Loss Exploding]
        Issue2[Loss Not Decreasing]
        Issue3[GPU Out of Memory]
        Issue4[Training Too Slow]
    end
    
    subgraph "Solutions"
        Sol1[Reduce Learning Rate<br/>Check gradient clipping]
        Sol2[Increase Learning Rate<br/>Check data quality<br/>Verify targets]
        Sol3[Reduce Batch Size<br/>Reduce Network Size<br/>Reduce Unroll Steps]
        Sol4[Increase Batch Size<br/>Reduce Checkpoint Frequency<br/>Optimize Data Loading]
    end
    
    Issue1 --> Sol1
    Issue2 --> Sol2
    Issue3 --> Sol3
    Issue4 --> Sol4
    
    style Issue1 fill:#FFB6C1
    style Issue2 fill:#FFB6C1
    style Issue3 fill:#FFB6C1
    style Issue4 fill:#FFB6C1
    style Sol1 fill:#90EE90
    style Sol2 fill:#90EE90
    style Sol3 fill:#90EE90
    style Sol4 fill:#90EE90
```

### Edge Case Handling

**1. Empty Replay Buffer**
```python
# Wait for games to be played
while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
    time.sleep(0.1)
```

**2. Termination Signal**
```python
# Check for early termination
while training_step < max_steps and not ray.get(
    shared_storage.get_info.remote("terminate")
):
    # Training loop
```

**3. Device Mismatch**
```python
# Warn if not training on GPU
if "cuda" not in str(next(self.model.parameters()).device):
    print("You are not training on GPU.\n")
```

**4. Optimizer State Loading**
```python
# Handle missing optimizer state
if initial_checkpoint["optimizer_state"] is not None:
    print("Loading optimizer...\n")
    self.optimizer.load_state_dict(initial_checkpoint["optimizer_state"])
```

**5. Gradient Numerical Stability**
```python
# Use cross-entropy instead of MSE for better stability
value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
```

---

## Best Practices

### Training Configuration

✅ **DO:**
- Start with default hyperparameters from paper
- Use GPU training for significant speedup
- Enable PER for sample efficiency
- Monitor all loss components separately
- Save checkpoints regularly
- Use exponential learning rate decay
- Balance training/self-play ratio

❌ **DON'T:**
- Train on CPU for large networks
- Use very large learning rates
- Ignore loss divergence warnings
- Skip checkpoint saving
- Set batch size too small (<32)
- Disable PER without good reason

### Debugging Training

**Monitor These Metrics:**
1. **Total Loss**: Should decrease over time
2. **Value Loss**: Should stabilize at reasonable level
3. **Reward Loss**: Should be small (rewards are sparse)
4. **Policy Loss**: Should decrease as policy improves
5. **Learning Rate**: Should decay smoothly
6. **Training/Self-Play Ratio**: Should stay near target

**Warning Signs:**
- Loss suddenly increases → reduce learning rate
- Loss plateaus early → increase learning rate or check data
- GPU utilization low → increase batch size
- Training much faster than self-play → adjust ratio

### Code Example: Custom Training Loop

```python
import ray
from knowledge.trainer import Trainer
from knowledge.shared_storage import SharedStorage
from knowledge.replay_buffer import ReplayBuffer

# Initialize Ray
ray.init()

# Create initial checkpoint
initial_checkpoint = {
    "weights": None,  # Will be initialized by network
    "optimizer_state": None,
    "training_step": 0,
    "num_played_games": 0,
    "num_played_steps": 0
}

# Create actors
shared_storage = SharedStorage.remote(initial_checkpoint, config)
replay_buffer = ReplayBuffer.remote(initial_checkpoint, {}, config)
trainer = Trainer.remote(initial_checkpoint, config)

# Start training
trainer.continuous_update_weights.remote(replay_buffer, shared_storage)

# Monitor progress
while True:
    info = ray.get(shared_storage.get_info.remote([
        "training_step", "total_loss", "value_loss"
    ]))
    print(f"Step: {info['training_step']}, "
          f"Loss: {info['total_loss']:.4f}, "
          f"Value: {info['value_loss']:.4f}")
    time.sleep(10)
```

---

## Related Modules

- **[Neural Networks](neural_networks.md)**: Network architectures trained by this module
- **[Replay Buffer](replay_buffer.md)**: Source of training batches and priority updates
- **[Shared Storage](shared_storage.md)**: Storage for model weights and training metrics
- **[Self-Play MCTS](self_play_mcts.md)**: Generates training data through game play
- **[Main Orchestrator](main_orchestrator.md)**: Coordinates trainer with other components

---

## References

### MuZero Paper
- **Title**: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
- **Authors**: Julian Schrittwieser et al.
- **Link**: https://arxiv.org/abs/1911.08265
- **Key Sections**: 
  - Appendix B: Training
  - Appendix D: Reanalyze
  - Appendix F: Hyperparameters

### Related Papers
- **Prioritized Experience Replay**: https://arxiv.org/abs/1511.05952
- **Distributed Prioritized Experience Replay**: https://arxiv.org/abs/1803.00933

### Implementation Details
- PyTorch Documentation: https://pytorch.org/docs/
- Ray Documentation: https://docs.ray.io/

---

## Summary

The **Trainer** module is the optimization engine of MuZero, implementing sophisticated training algorithms that enable the system to learn effective value, policy, and dynamics models from self-play experience. Key features include:

- **Continuous Training**: Runs concurrently with self-play in a dedicated thread
- **Multi-Step Unrolling**: Trains dynamics model through recurrent inference
- **Multi-Head Losses**: Simultaneously optimizes value, reward, and policy predictions
- **Prioritized Replay**: Focuses learning on high-error transitions
- **Gradient Management**: Careful scaling for stable training
- **Flexible Configuration**: Extensive hyperparameters for tuning

The module's design balances training efficiency, sample efficiency, and numerical stability, making it a critical component in MuZero's ability to master complex domains through self-play reinforcement learning.
