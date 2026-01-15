# Replay Buffer Module

## Overview

The **replay_buffer** module is a critical component of the MuZero reinforcement learning system that manages experience storage and batch generation for training. It implements two key distributed actors using Ray: **ReplayBuffer** for storing and sampling game experiences, and **Reanalyse** for updating stored experiences with fresh model predictions. This module enables efficient experience replay with support for Prioritized Experience Replay (PER) and implements the reanalysis mechanism described in the MuZero paper.

### Key Responsibilities

- **Experience Storage**: Maintains a buffer of completed game histories with configurable size limits
- **Batch Generation**: Samples and prepares training batches from stored experiences
- **Prioritized Sampling**: Implements PER to focus training on more valuable experiences
- **Experience Reanalysis**: Updates stored experiences with fresh model predictions for improved value estimates
- **Distributed Operation**: Runs as independent Ray actors for concurrent operation with training and self-play

### Module Position in System Architecture

```mermaid
graph TB
    subgraph "MuZero System Architecture"
        SP[Self-Play MCTS<br/>Game Generation]
        RB[Replay Buffer<br/>Experience Storage]
        RA[Reanalyse<br/>Experience Update]
        TR[Trainer<br/>Network Training]
        SS[Shared Storage<br/>Weights & Metrics]
        NN[Neural Networks<br/>Model Inference]
    end
    
    SP -->|Save Game History| RB
    RB -->|Sample Batch| TR
    TR -->|Update Priorities| RB
    TR -->|Save Weights| SS
    SS -->|Get Weights| RA
    RB -->|Sample Game| RA
    RA -->|Update History| RB
    RA -->|Use Model| NN
    SS -->|Provide Stats| SP
    
    style RB fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style RA fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    
    classDef current fill:#e1f5ff,stroke:#01579b,stroke-width:3px
```

---

## Core Components

### 1. ReplayBuffer

A Ray remote actor that manages the storage of game histories and generates training batches.

#### Architecture

```mermaid
classDiagram
    class ReplayBuffer {
        -config: Config
        -buffer: Dict[int, GameHistory]
        -num_played_games: int
        -num_played_steps: int
        -total_samples: int
        
        +__init__(initial_checkpoint, initial_buffer, config)
        +save_game(game_history, shared_storage)
        +get_buffer() Dict
        +get_batch() Tuple
        +sample_game(force_uniform) Tuple
        +sample_n_games(n_games, force_uniform) List
        +sample_position(game_history, force_uniform) Tuple
        +update_game_history(game_id, game_history)
        +update_priorities(priorities, index_info)
        +compute_target_value(game_history, index) float
        +make_target(game_history, state_index) Tuple
    }
    
    class GameHistory {
        +observation_history: List
        +action_history: List
        +reward_history: List
        +root_values: List
        +child_visits: List
        +priorities: ndarray
        +game_priority: float
        +reanalysed_predicted_root_values: ndarray
        +get_stacked_observations()
    }
    
    class SharedStorage {
        +set_info(key, value)
        +get_info(key)
    }
    
    ReplayBuffer --> GameHistory : stores
    ReplayBuffer --> SharedStorage : updates stats
```

#### Key Features

**Experience Storage Management**
- Maintains a dictionary buffer mapping game IDs to GameHistory objects
- Implements FIFO eviction when buffer reaches capacity
- Tracks global statistics (games played, steps taken, total samples)
- Supports initialization from checkpoints for training resumption

**Prioritized Experience Replay (PER)**
- Assigns priorities based on TD-error magnitude: `priority = |value - target|^α`
- Maintains both position-level and game-level priorities
- Samples games and positions according to priority distribution
- Computes importance sampling weights to correct for bias

**Batch Generation Pipeline**
1. Sample N games from buffer (prioritized or uniform)
2. For each game, sample a position (prioritized or uniform)
3. Generate targets for unroll steps from that position
4. Stack observations with action history
5. Compute importance sampling weights
6. Return structured batch for training

#### Data Flow

```mermaid
sequenceDiagram
    participant SP as Self-Play
    participant RB as ReplayBuffer
    participant TR as Trainer
    participant SS as SharedStorage
    
    SP->>RB: save_game(game_history)
    activate RB
    RB->>RB: Compute initial priorities (PER)
    RB->>RB: Add to buffer
    RB->>RB: Evict old games if full
    RB->>SS: Update num_played_games/steps
    deactivate RB
    
    TR->>RB: get_batch()
    activate RB
    RB->>RB: sample_n_games()
    loop For each game
        RB->>RB: sample_position()
        RB->>RB: make_target()
        RB->>RB: get_stacked_observations()
    end
    RB-->>TR: Return batch with weights
    deactivate RB
    
    TR->>RB: update_priorities(new_priorities)
    activate RB
    RB->>RB: Update position priorities
    RB->>RB: Update game priorities
    deactivate RB
```

#### Methods Deep Dive

**save_game(game_history, shared_storage)**
- Computes initial priorities for PER if enabled
- Adds game to buffer with auto-incrementing game ID
- Removes oldest game if buffer exceeds `replay_buffer_size`
- Updates global statistics in shared storage

**get_batch()**
Returns a tuple of `(index_batch, data_batch)` where:
- `index_batch`: List of `[game_id, position]` for priority updates
- `data_batch`: Tuple of:
  - `observation_batch`: Stacked observations with action history
  - `action_batch`: Actions taken for unroll steps
  - `value_batch`: Target values for each step
  - `reward_batch`: Observed rewards
  - `policy_batch`: MCTS policy distributions
  - `weight_batch`: Importance sampling weights (PER only)
  - `gradient_scale_batch`: Gradient scaling factors

**sample_game(force_uniform=False)**
- Samples a game from buffer
- Uses priority-based sampling if PER enabled and not forced uniform
- Returns `(game_id, game_history, game_probability)`

**sample_position(game_history, force_uniform=False)**
- Samples a position within a game
- Uses position priorities if PER enabled
- Returns `(position_index, position_probability)`

**compute_target_value(game_history, index)**
Implements n-step bootstrapped value target:
```
V_target = Σ(γ^i * r_{t+i}) + γ^n * V(s_{t+n})
```
- Sums discounted rewards for `td_steps`
- Adds discounted bootstrap value from search tree
- Handles player perspective in two-player games
- Returns 0 for terminal states

**make_target(game_history, state_index)**
- Generates targets for `num_unroll_steps + 1` positions
- For positions within game: uses actual values, rewards, policies, actions
- For terminal position: uses 0 value, actual reward, uniform policy
- For absorbing states: uses 0 value/reward, uniform policy, random action

**update_priorities(priorities, index_info)**
- Updates position priorities from training TD-errors
- Recomputes game priority as max of position priorities
- Handles cases where games were evicted during training

---

### 2. Reanalyse

A Ray remote actor that continuously updates stored game histories with fresh model predictions.

#### Architecture

```mermaid
classDiagram
    class Reanalyse {
        -config: Config
        -model: MuZeroNetwork
        -num_reanalysed_games: int
        
        +__init__(initial_checkpoint, config)
        +reanalyse(replay_buffer, shared_storage)
    }
    
    class MuZeroNetwork {
        +initial_inference(observation)
        +set_weights(weights)
        +eval()
    }
    
    class ReplayBuffer {
        +sample_game(force_uniform)
        +update_game_history(game_id, game_history)
    }
    
    class SharedStorage {
        +get_info(key)
        +set_info(key, value)
    }
    
    Reanalyse --> MuZeroNetwork : uses
    Reanalyse --> ReplayBuffer : samples & updates
    Reanalyse --> SharedStorage : syncs weights
```

#### Purpose and Benefits

The reanalysis mechanism addresses a key challenge in off-policy learning: stored experiences contain value estimates from older, less accurate models. By re-evaluating positions with the current model, reanalysis provides:

1. **Fresher Value Estimates**: More accurate n-step bootstrap values
2. **Stable Learning**: Reduces bias from outdated predictions
3. **Improved Sample Efficiency**: Extracts more value from stored experiences
4. **Better Credit Assignment**: More accurate value targets for training

#### Reanalysis Process

```mermaid
sequenceDiagram
    participant RA as Reanalyse
    participant SS as SharedStorage
    participant RB as ReplayBuffer
    participant NN as MuZeroNetwork
    
    loop Continuous Reanalysis
        RA->>SS: get_info("weights")
        SS-->>RA: Latest model weights
        RA->>NN: set_weights(weights)
        
        RA->>RB: sample_game(force_uniform=True)
        RB-->>RA: game_history
        
        RA->>RA: Extract all observations
        RA->>NN: initial_inference(observations)
        NN-->>RA: Predicted values
        
        RA->>RA: Update reanalysed_predicted_root_values
        RA->>RB: update_game_history(game_id, game_history)
        
        RA->>SS: set_info("num_reanalysed_games", count)
    end
```

#### Key Features

**Continuous Operation**
- Runs in infinite loop until training completes
- Waits for at least one game to be played before starting
- Checks termination flag from shared storage

**Model Synchronization**
- Loads latest weights from shared storage each iteration
- Runs on GPU if `reanalyse_on_gpu` is enabled
- Uses model in evaluation mode (no gradient computation)

**Uniform Sampling**
- Always samples games uniformly (not by priority)
- Ensures all experiences eventually get reanalyzed
- Prevents bias toward high-priority games

**Batch Inference**
- Processes all positions in a game simultaneously
- Converts observations to stacked format
- Stores predictions in `reanalysed_predicted_root_values`

#### Configuration Options

Key configuration parameters affecting reanalysis:

- `use_last_model_value`: Enable/disable reanalysis mechanism
- `reanalyse_on_gpu`: Run reanalysis on GPU for faster inference
- `stacked_observations`: Number of past observations to stack
- `support_size`: Size of value support for categorical distribution

---

## Integration with Other Modules

### Relationship with Self-Play MCTS

```mermaid
graph LR
    subgraph "Self-Play Process"
        MCTS[MCTS Search]
        GH[GameHistory]
    end
    
    subgraph "Replay Buffer"
        RB[ReplayBuffer]
        PER[Priority Computation]
    end
    
    MCTS -->|Store search stats| GH
    GH -->|Complete game| RB
    RB -->|Compute TD-error| PER
    PER -->|Assign priorities| GH
    
    style GH fill:#fff3e0,stroke:#e65100
    style RB fill:#e1f5ff,stroke:#01579b
```

**Data Flow**: Self-play generates complete GameHistory objects containing:
- Observation sequences
- Action sequences  
- Reward sequences
- MCTS policy distributions (child_visits)
- Root value estimates

See [self_play_mcts.md](self_play_mcts.md) for details on game history generation.

### Relationship with Trainer

```mermaid
graph TB
    subgraph "Training Loop"
        TR[Trainer]
        LOSS[Loss Computation]
        GRAD[Gradient Update]
    end
    
    subgraph "Replay Buffer"
        RB[ReplayBuffer]
        BATCH[Batch Generation]
    end
    
    TR -->|Request batch| RB
    RB -->|Sample & prepare| BATCH
    BATCH -->|Return batch + weights| TR
    TR -->|Forward pass| LOSS
    LOSS -->|Compute TD-errors| GRAD
    GRAD -->|Update priorities| RB
    
    style TR fill:#f3e5f5,stroke:#4a148c
    style RB fill:#e1f5ff,stroke:#01579b
```

**Bidirectional Communication**:
- Trainer requests batches for training
- Replay buffer provides weighted samples
- Trainer returns TD-errors for priority updates

See [trainer.md](trainer.md) for training loop details.

### Relationship with Shared Storage

```mermaid
graph LR
    subgraph "Replay Buffer"
        RB[ReplayBuffer]
        RA[Reanalyse]
    end
    
    subgraph "Shared Storage"
        SS[SharedStorage]
        STATS[Statistics]
        WEIGHTS[Model Weights]
    end
    
    RB -->|Update game/step counts| STATS
    RA -->|Update reanalysis count| STATS
    RA -->|Fetch latest weights| WEIGHTS
    
    style RB fill:#e1f5ff,stroke:#01579b
    style RA fill:#e1f5ff,stroke:#01579b
    style SS fill:#e8f5e9,stroke:#1b5e20
```

**Coordination**: Shared storage acts as central coordination point for:
- Training progress tracking
- Model weight distribution
- Termination signaling

See [shared_storage.md](shared_storage.md) for coordination mechanisms.

### Relationship with Neural Networks

```mermaid
graph TB
    subgraph "Reanalyse"
        RA[Reanalyse Actor]
        OBS[Observation Batch]
    end
    
    subgraph "Neural Networks"
        MZ[MuZeroNetwork]
        REP[Representation Network]
        PRED[Prediction Network]
    end
    
    RA -->|Prepare observations| OBS
    OBS -->|initial_inference| MZ
    MZ -->|Encode| REP
    REP -->|Predict value| PRED
    PRED -->|Return values| RA
    
    style RA fill:#e1f5ff,stroke:#01579b
    style MZ fill:#fce4ec,stroke:#880e4f
```

**Model Usage**: Reanalyse uses the representation and prediction networks to generate fresh value estimates without performing MCTS search.

See [neural_networks.md](neural_networks.md) for network architecture details.

---

## Key Algorithms

### Prioritized Experience Replay (PER)

The module implements PER as described in "Prioritized Experience Replay" (Schaul et al., 2015) with adaptations for MuZero.

#### Priority Computation

```mermaid
flowchart TD
    START[New Game Saved] --> CHECK{PER Enabled?}
    CHECK -->|No| STORE[Store with uniform priority]
    CHECK -->|Yes| COMPUTE[Compute Position Priorities]
    
    COMPUTE --> LOOP[For each position i]
    LOOP --> VALUE[Get root_value_i]
    LOOP --> TARGET[Compute target_value_i]
    LOOP --> DELTA["δ_i = abs root_value_i - target_value_i"]
    LOOP --> PRIORITY["priority_i = δ_i power α"]
    
    PRIORITY --> GAME_PRI["game_priority = max priorities"]
    GAME_PRI --> STORE
    STORE --> END[Add to Buffer]
    
    style COMPUTE fill:#fff3e0
    style PRIORITY fill:#e1f5ff
    style GAME_PRI fill:#e1f5ff
```

**Priority Formula**:
```
position_priority = |TD_error|^α
game_priority = max(position_priorities)
```

Where:
- `TD_error = root_value - target_value`
- `α` (PER_alpha): Controls prioritization strength (typically 0.5-1.0)

#### Sampling with Priorities

```mermaid
flowchart TD
    START[Sample Batch] --> GAME_SAMPLE[Sample N Games]
    GAME_SAMPLE --> GAME_PROB[P_game_i ∝ game_priority_i]
    
    GAME_PROB --> POS_SAMPLE[For each game: Sample Position]
    POS_SAMPLE --> POS_PROB[P_pos_j ∝ position_priority_j]
    
    POS_PROB --> WEIGHT[Compute IS Weight]
    WEIGHT --> FORMULA[w_i = 1 / N * P_game * P_pos]
    FORMULA --> NORMALIZE[Normalize: w_i = w_i / max_w]
    
    NORMALIZE --> RETURN[Return Batch with Weights]
    
    style GAME_PROB fill:#e1f5ff
    style POS_PROB fill:#e1f5ff
    style NORMALIZE fill:#fff3e0
```

**Importance Sampling Weight**:
```
w = (1 / (N * P_game * P_position))
normalized_w = w / max(weights)
```

This corrects for bias introduced by non-uniform sampling.

#### Priority Updates

After training, priorities are updated based on fresh TD-errors:

```python
# During training
td_errors = |predicted_values - target_values|
new_priorities = td_errors ** α

# Update in replay buffer
replay_buffer.update_priorities(new_priorities, index_info)
```

### Target Value Computation

The n-step bootstrapped return combines observed rewards with model predictions.

#### Algorithm

```mermaid
flowchart TD
    START[Compute Target Value at t] --> INIT[value = 0]
    INIT --> BOOTSTRAP{t + n < game_length?}
    
    BOOTSTRAP -->|Yes| GET_ROOT[Get root_value at t+n]
    GET_ROOT --> REANALYSE{Reanalysed values exist?}
    REANALYSE -->|Yes| USE_REANALYSE[Use reanalysed_value]
    REANALYSE -->|No| USE_ORIGINAL[Use original root_value]
    
    USE_REANALYSE --> PERSPECTIVE{Same player?}
    USE_ORIGINAL --> PERSPECTIVE
    PERSPECTIVE -->|Yes| POSITIVE[value = +root_value * γ^n]
    PERSPECTIVE -->|No| NEGATIVE[value = -root_value * γ^n]
    
    BOOTSTRAP -->|No| TERMINAL[value = 0]
    
    POSITIVE --> REWARDS[Add Discounted Rewards]
    NEGATIVE --> REWARDS
    TERMINAL --> REWARDS
    
    REWARDS --> LOOP[For i in 0 to n-1]
    LOOP --> REWARD_PERSP{Same player at t+i?}
    REWARD_PERSP -->|Yes| ADD_POS[value += reward * γ^i]
    REWARD_PERSP -->|No| ADD_NEG[value -= reward * γ^i]
    
    ADD_POS --> RETURN[Return value]
    ADD_NEG --> RETURN
    
    style USE_REANALYSE fill:#e1f5ff
    style REWARDS fill:#fff3e0
```

#### Mathematical Formulation

For single-player games:
```
V_target(s_t) = Σ_{i=0}^{n-1} γ^i * r_{t+i+1} + γ^n * V(s_{t+n})
```

For two-player games (player perspective):
```
V_target(s_t) = Σ_{i=0}^{n-1} γ^i * sign(player_t, player_{t+i}) * r_{t+i+1} 
                + γ^n * sign(player_t, player_{t+n}) * V(s_{t+n})

where sign(p1, p2) = +1 if p1 == p2, else -1
```

#### Reanalysis Integration

When reanalysis is enabled:
```python
if game_history.reanalysed_predicted_root_values is not None:
    root_values = game_history.reanalysed_predicted_root_values
else:
    root_values = game_history.root_values
```

This ensures the most recent value estimates are used for bootstrapping.

### Batch Generation Process

Complete pipeline for generating a training batch:

```mermaid
flowchart TD
    START[get_batch Called] --> INIT[Initialize Empty Lists]
    INIT --> SAMPLE_GAMES[Sample batch_size Games]
    
    SAMPLE_GAMES --> LOOP_START{For each game}
    LOOP_START --> SAMPLE_POS[Sample Position in Game]
    SAMPLE_POS --> MAKE_TARGET[Generate Targets for Unroll Steps]
    
    MAKE_TARGET --> STACK_OBS[Stack Observations with Actions]
    STACK_OBS --> COMPUTE_SCALE[Compute Gradient Scale]
    COMPUTE_SCALE --> COMPUTE_WEIGHT{PER Enabled?}
    
    COMPUTE_WEIGHT -->|Yes| CALC_WEIGHT[w = 1 / N * P_game * P_pos]
    COMPUTE_WEIGHT -->|No| SKIP_WEIGHT[weight = None]
    
    CALC_WEIGHT --> APPEND[Append to Batch Lists]
    SKIP_WEIGHT --> APPEND
    
    APPEND --> LOOP_END{More games?}
    LOOP_END -->|Yes| LOOP_START
    LOOP_END -->|No| NORMALIZE{PER Enabled?}
    
    NORMALIZE -->|Yes| NORM_WEIGHTS[Normalize Weights by Max]
    NORMALIZE -->|No| RETURN
    
    NORM_WEIGHTS --> RETURN[Return index_batch, data_batch]
    
    style SAMPLE_GAMES fill:#e1f5ff
    style MAKE_TARGET fill:#fff3e0
    style CALC_WEIGHT fill:#e1f5ff
```

#### Gradient Scale Computation

The gradient scale determines how many valid unroll steps exist:

```python
gradient_scale = min(
    num_unroll_steps,
    len(game_history.action_history) - game_pos
)
```

This is used during training to mask out invalid positions beyond game termination.

---

## Configuration Parameters

### Replay Buffer Configuration

| Parameter | Type | Description | Typical Value |
|-----------|------|-------------|---------------|
| `replay_buffer_size` | int | Maximum number of games to store | 1000-10000 |
| `batch_size` | int | Number of samples per training batch | 128-1024 |
| `num_unroll_steps` | int | Number of future steps to unroll | 5-10 |
| `td_steps` | int | Number of steps for n-step return | 5-50 |
| `discount` | float | Discount factor γ | 0.95-0.999 |
| `stacked_observations` | int | Number of past observations to stack | 1-8 |
| `action_space` | list | List of valid actions | Game-specific |
| `seed` | int | Random seed for reproducibility | Any int |

### Prioritized Experience Replay Configuration

| Parameter | Type | Description | Typical Value |
|-----------|------|-------------|---------------|
| `PER` | bool | Enable prioritized experience replay | True/False |
| `PER_alpha` | float | Priority exponent (α) | 0.5-1.0 |

Higher α values increase prioritization strength:
- α = 0: Uniform sampling
- α = 1: Full prioritization

### Reanalysis Configuration

| Parameter | Type | Description | Typical Value |
|-----------|------|-------------|---------------|
| `use_last_model_value` | bool | Enable reanalysis mechanism | True |
| `reanalyse_on_gpu` | bool | Run reanalysis on GPU | True |
| `training_steps` | int | Total training steps (for termination) | 100000-1000000 |

---

## Performance Considerations

### Memory Management

```mermaid
graph TB
    subgraph "Memory Usage"
        BUFFER[Buffer Dictionary]
        GAMES[Game Histories]
        OBS[Observations]
        PRIORITIES[Priority Arrays]
    end
    
    subgraph "Memory Control"
        SIZE[replay_buffer_size]
        EVICT[FIFO Eviction]
        TRACK[total_samples Counter]
    end
    
    BUFFER --> GAMES
    GAMES --> OBS
    GAMES --> PRIORITIES
    
    SIZE --> EVICT
    EVICT --> BUFFER
    TRACK --> BUFFER
    
    style BUFFER fill:#ffebee,stroke:#c62828
    style EVICT fill:#e8f5e9,stroke:#2e7d32
```

**Memory Footprint**:
```
Total Memory ≈ replay_buffer_size × avg_game_length × observation_size
```

**Optimization Strategies**:
1. **Limit Buffer Size**: Set `replay_buffer_size` based on available RAM
2. **FIFO Eviction**: Automatically removes oldest games
3. **Lazy Loading**: Consider disk-based storage for very large buffers
4. **Compression**: Store observations in compressed format

### Computational Efficiency

#### Batch Generation Performance

```mermaid
graph LR
    subgraph "Bottlenecks"
        SAMPLE[Game Sampling]
        TARGET[Target Computation]
        STACK[Observation Stacking]
    end
    
    subgraph "Optimizations"
        NUMPY[NumPy Vectorization]
        CACHE[Priority Caching]
        PARALLEL[Ray Parallelism]
    end
    
    SAMPLE --> NUMPY
    TARGET --> NUMPY
    STACK --> NUMPY
    
    SAMPLE --> CACHE
    SAMPLE --> PARALLEL
    
    style NUMPY fill:#e8f5e9,stroke:#2e7d32
    style CACHE fill:#e8f5e9,stroke:#2e7d32
    style PARALLEL fill:#e8f5e9,stroke:#2e7d32
```

**Performance Tips**:
1. **Vectorized Operations**: Use NumPy for array operations
2. **Priority Caching**: Cache game priorities for faster sampling
3. **Batch Processing**: Process multiple samples simultaneously
4. **Ray Distribution**: Leverage Ray's distributed computing

#### Reanalysis Performance

**GPU Acceleration**:
- Set `reanalyse_on_gpu=True` for 5-10x speedup
- Batch all positions in a game for efficient inference
- Use mixed precision if supported

**Throughput Optimization**:
```python
# Reanalysis rate should match or exceed game generation rate
reanalysis_rate ≈ game_generation_rate

# Monitor with:
num_reanalysed_games / num_played_games
```

### Distributed Coordination

```mermaid
sequenceDiagram
    participant SP as Self-Play (N actors)
    participant RB as ReplayBuffer (1 actor)
    participant RA as Reanalyse (1 actor)
    participant TR as Trainer (1 actor)
    
    par Concurrent Operations
        loop Self-Play
            SP->>RB: save_game()
        end
    and
        loop Training
            TR->>RB: get_batch()
            TR->>RB: update_priorities()
        end
    and
        loop Reanalysis
            RA->>RB: sample_game()
            RA->>RB: update_game_history()
        end
    end
    
    Note over SP,TR: All operations are non-blocking via Ray
```

**Concurrency Benefits**:
- Self-play, training, and reanalysis run in parallel
- Ray handles synchronization automatically
- No GIL limitations due to separate processes

**Potential Bottlenecks**:
- Replay buffer can become a bottleneck if accessed too frequently
- Consider multiple replay buffer actors for extreme scale
- Monitor Ray dashboard for actor utilization

---

## Usage Examples

### Basic Initialization

```python
import ray
from replay_buffer import ReplayBuffer, Reanalyse

# Initialize Ray
ray.init()

# Create replay buffer
initial_checkpoint = {
    "num_played_games": 0,
    "num_played_steps": 0,
    "num_reanalysed_games": 0,
    "weights": model.get_weights()
}
initial_buffer = {}

replay_buffer = ReplayBuffer.remote(
    initial_checkpoint,
    initial_buffer,
    config
)

# Create reanalyse actor
reanalyse = Reanalyse.remote(initial_checkpoint, config)
```

### Saving Game Histories

```python
# After self-play game completes
game_history = GameHistory()
# ... populate game_history during self-play ...

# Save to replay buffer
ray.get(replay_buffer.save_game.remote(game_history, shared_storage))
```

### Sampling Training Batches

```python
# In training loop
index_batch, data_batch = ray.get(replay_buffer.get_batch.remote())

(
    observation_batch,
    action_batch,
    value_batch,
    reward_batch,
    policy_batch,
    weight_batch,
    gradient_scale_batch
) = data_batch

# Use batch for training
loss = train_step(
    model,
    observation_batch,
    action_batch,
    value_batch,
    reward_batch,
    policy_batch,
    weight_batch
)
```

### Updating Priorities

```python
# After computing TD-errors during training
td_errors = compute_td_errors(predictions, targets)
priorities = td_errors ** config.PER_alpha

# Update priorities in replay buffer
replay_buffer.update_priorities.remote(priorities, index_batch)
```

### Running Reanalysis

```python
# Start reanalysis in background
reanalyse_task = reanalyse.reanalyse.remote(replay_buffer, shared_storage)

# Reanalysis runs continuously until training completes
# No need to wait for completion
```

### Monitoring Statistics

```python
# Get buffer statistics
buffer = ray.get(replay_buffer.get_buffer.remote())
print(f"Buffer size: {len(buffer)} games")
print(f"Total samples: {sum(len(g.root_values) for g in buffer.values())}")

# Get reanalysis progress
num_reanalysed = ray.get(shared_storage.get_info.remote("num_reanalysed_games"))
num_played = ray.get(shared_storage.get_info.remote("num_played_games"))
print(f"Reanalysis coverage: {num_reanalysed / num_played * 100:.1f}%")
```

---

## Error Handling and Edge Cases

### Buffer Eviction During Training

**Problem**: A game may be evicted from the buffer between sampling and priority update.

**Solution**: Check if game still exists before updating:
```python
def update_priorities(self, priorities, index_info):
    for i in range(len(index_info)):
        game_id, game_pos = index_info[i]
        
        # Check if game still in buffer
        if next(iter(self.buffer)) <= game_id:
            # Safe to update
            self.buffer[game_id].priorities[game_pos:...] = priorities[i, :]
```

### Absorbing States Beyond Game End

**Problem**: Unroll steps may extend beyond game termination.

**Solution**: Treat post-terminal states as absorbing states:
```python
if current_index > len(game_history.root_values):
    # Absorbing state
    target_values.append(0)
    target_rewards.append(0)
    target_policies.append(uniform_policy)
    actions.append(random_action)
```

### Reanalysis Synchronization

**Problem**: Reanalysis may use stale weights if not synchronized properly.

**Solution**: Fetch latest weights at start of each reanalysis iteration:
```python
def reanalyse(self, replay_buffer, shared_storage):
    while not terminated:
        # Always use latest weights
        self.model.set_weights(
            ray.get(shared_storage.get_info.remote("weights"))
        )
        # ... perform reanalysis ...
```

### Priority Initialization

**Problem**: New games need initial priorities before training.

**Solution**: Compute priorities based on initial TD-error:
```python
priorities = []
for i, root_value in enumerate(game_history.root_values):
    target_value = self.compute_target_value(game_history, i)
    priority = (abs(root_value - target_value) ** self.config.PER_alpha)
    priorities.append(priority)
```

### Empty Buffer Handling

**Problem**: Training may start before any games are played.

**Solution**: Wait for minimum buffer size:
```python
# In trainer
while ray.get(replay_buffer.get_buffer.remote()) < min_buffer_size:
    time.sleep(1)
```

---

## Testing and Validation

### Unit Tests

```python
def test_replay_buffer_capacity():
    """Test FIFO eviction when buffer is full"""
    config.replay_buffer_size = 10
    replay_buffer = ReplayBuffer(checkpoint, {}, config)
    
    # Add 15 games
    for i in range(15):
        game_history = create_dummy_game()
        replay_buffer.save_game(game_history)
    
    buffer = replay_buffer.get_buffer()
    assert len(buffer) == 10
    assert min(buffer.keys()) == 5  # First 5 games evicted

def test_priority_sampling():
    """Test that high-priority games are sampled more often"""
    replay_buffer = create_buffer_with_priorities()
    
    sample_counts = defaultdict(int)
    for _ in range(1000):
        game_id, _, _ = replay_buffer.sample_game()
        sample_counts[game_id] += 1
    
    # High priority game should be sampled more
    assert sample_counts[high_priority_game_id] > sample_counts[low_priority_game_id]

def test_target_value_computation():
    """Test n-step return calculation"""
    game_history = create_test_game()
    replay_buffer = ReplayBuffer(checkpoint, {}, config)
    
    target = replay_buffer.compute_target_value(game_history, index=0)
    
    # Manually compute expected value
    expected = sum(
        config.discount**i * game_history.reward_history[i+1]
        for i in range(config.td_steps)
    ) + config.discount**config.td_steps * game_history.root_values[config.td_steps]
    
    assert abs(target - expected) < 1e-5
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete replay buffer pipeline"""
    # Initialize components
    replay_buffer = ReplayBuffer.remote(checkpoint, {}, config)
    shared_storage = SharedStorage.remote(checkpoint, config)
    
    # Generate and save games
    for _ in range(10):
        game_history = play_game()
        ray.get(replay_buffer.save_game.remote(game_history, shared_storage))
    
    # Sample batch
    index_batch, data_batch = ray.get(replay_buffer.get_batch.remote())
    
    # Verify batch structure
    assert len(index_batch) == config.batch_size
    assert len(data_batch) == 7  # 7 components
    
    # Verify shapes
    obs_batch, action_batch, value_batch, _, _, _, _ = data_batch
    assert len(obs_batch) == config.batch_size
    assert len(action_batch[0]) == config.num_unroll_steps + 1

def test_reanalysis_updates():
    """Test that reanalysis updates game histories"""
    replay_buffer = ReplayBuffer.remote(checkpoint, {}, config)
    reanalyse = Reanalyse.remote(checkpoint, config)
    shared_storage = SharedStorage.remote(checkpoint, config)
    
    # Add game
    game_history = play_game()
    ray.get(replay_buffer.save_game.remote(game_history, shared_storage))
    
    # Run one reanalysis iteration
    original_values = game_history.root_values.copy()
    
    # Trigger reanalysis (run one iteration manually)
    game_id, game_history, _ = ray.get(
        replay_buffer.sample_game.remote(force_uniform=True)
    )
    # ... perform reanalysis ...
    ray.get(replay_buffer.update_game_history.remote(game_id, game_history))
    
    # Verify reanalysed values exist
    updated_game = ray.get(replay_buffer.get_buffer.remote())[game_id]
    assert updated_game.reanalysed_predicted_root_values is not None
```

### Performance Benchmarks

```python
def benchmark_batch_generation():
    """Measure batch generation throughput"""
    replay_buffer = create_full_buffer()
    
    start_time = time.time()
    num_batches = 100
    
    for _ in range(num_batches):
        ray.get(replay_buffer.get_batch.remote())
    
    elapsed = time.time() - start_time
    throughput = num_batches / elapsed
    
    print(f"Batch generation: {throughput:.2f} batches/sec")
    assert throughput > 10  # Should generate at least 10 batches/sec

def benchmark_reanalysis():
    """Measure reanalysis throughput"""
    reanalyse = Reanalyse.remote(checkpoint, config)
    replay_buffer = create_full_buffer()
    shared_storage = SharedStorage.remote(checkpoint, config)
    
    start_time = time.time()
    num_games = 50
    
    for _ in range(num_games):
        game_id, game_history, _ = ray.get(
            replay_buffer.sample_game.remote(force_uniform=True)
        )
        # ... perform reanalysis ...
    
    elapsed = time.time() - start_time
    throughput = num_games / elapsed
    
    print(f"Reanalysis: {throughput:.2f} games/sec")
```

---

## Troubleshooting Guide

### Common Issues

#### Issue: Out of Memory Errors

**Symptoms**:
- Process killed by OS
- Ray actor crashes
- Slow performance due to swapping

**Solutions**:
1. Reduce `replay_buffer_size`
2. Reduce `stacked_observations`
3. Use smaller observation representations
4. Enable disk-based storage for large buffers

```python
# Monitor memory usage
import psutil
process = psutil.Process()
memory_gb = process.memory_info().rss / 1e9
print(f"Replay buffer memory: {memory_gb:.2f} GB")
```

#### Issue: Slow Batch Generation

**Symptoms**:
- Training waits for batches
- Low GPU utilization
- High replay buffer CPU usage

**Solutions**:
1. Reduce `batch_size`
2. Optimize `compute_target_value` with vectorization
3. Cache frequently accessed data
4. Use multiple replay buffer actors

```python
# Profile batch generation
import cProfile
cProfile.run('replay_buffer.get_batch()', sort='cumtime')
```

#### Issue: Reanalysis Falling Behind

**Symptoms**:
- `num_reanalysed_games << num_played_games`
- Reanalysis actor at 100% CPU/GPU
- Stale value estimates

**Solutions**:
1. Enable `reanalyse_on_gpu=True`
2. Reduce model size for faster inference
3. Run multiple reanalysis actors
4. Increase reanalysis batch size

```python
# Monitor reanalysis coverage
coverage = num_reanalysed_games / num_played_games
if coverage < 0.5:
    print("WARNING: Reanalysis falling behind!")
```

#### Issue: Priority Sampling Bias

**Symptoms**:
- Same games sampled repeatedly
- Poor exploration of buffer
- Training instability

**Solutions**:
1. Reduce `PER_alpha` for less aggressive prioritization
2. Add minimum priority threshold
3. Periodically sample uniformly
4. Implement priority decay

```python
# Add minimum priority
priority = max(min_priority, abs(td_error) ** alpha)

# Mix uniform and prioritized sampling
if random.random() < 0.1:
    game_id, game_history, _ = replay_buffer.sample_game(force_uniform=True)
```

### Debugging Tips

**Enable Verbose Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add logging to replay buffer
print(f"Buffer size: {len(self.buffer)}")
print(f"Total samples: {self.total_samples}")
print(f"Priority range: [{min(priorities)}, {max(priorities)}]")
```

**Visualize Priority Distribution**:
```python
import matplotlib.pyplot as plt

priorities = [game.game_priority for game in buffer.values()]
plt.hist(priorities, bins=50)
plt.xlabel("Game Priority")
plt.ylabel("Count")
plt.title("Priority Distribution")
plt.show()
```

**Monitor Ray Dashboard**:
- Open Ray dashboard at `http://localhost:8265`
- Check actor resource utilization
- Monitor task execution times
- Identify bottlenecks

---

## Best Practices

### Configuration Tuning

1. **Buffer Size**: Start with 1000 games, increase if memory allows
2. **Batch Size**: Balance between sample diversity and training speed
3. **Unroll Steps**: Longer unrolls improve credit assignment but increase computation
4. **TD Steps**: Match to game episode length (5-10 for short games, 20-50 for long games)
5. **PER Alpha**: Start with 0.6, increase for more prioritization

### Memory Optimization

1. **Use Efficient Data Types**: Store observations as uint8 instead of float32
2. **Compress Observations**: Use image compression for visual observations
3. **Lazy Loading**: Load observations from disk when needed
4. **Periodic Cleanup**: Remove unnecessary data from game histories

### Performance Optimization

1. **Vectorize Operations**: Use NumPy/PyTorch for batch operations
2. **GPU Acceleration**: Enable GPU for reanalysis
3. **Parallel Sampling**: Sample multiple games concurrently
4. **Cache Priorities**: Avoid recomputing priorities unnecessarily

### Monitoring and Maintenance

1. **Track Key Metrics**:
   - Buffer utilization
   - Reanalysis coverage
   - Priority distribution
   - Batch generation time

2. **Regular Checkpointing**:
   ```python
   # Save buffer periodically
   if training_step % 1000 == 0:
       buffer = ray.get(replay_buffer.get_buffer.remote())
       torch.save(buffer, f"buffer_step_{training_step}.pkl")
   ```

3. **Validate Data Quality**:
   ```python
   # Check for NaN values
   assert not numpy.isnan(observation_batch).any()
   assert not numpy.isnan(value_batch).any()
   ```

---

## Future Enhancements

### Potential Improvements

1. **Distributed Replay Buffer**: Shard buffer across multiple actors for scalability
2. **Disk-Based Storage**: Support for buffers larger than RAM
3. **Advanced Prioritization**: Implement diversity-based sampling
4. **Compression**: Reduce memory footprint with observation compression
5. **Incremental Reanalysis**: Only reanalyze positions with large value changes
6. **Multi-Step Reanalysis**: Use recurrent inference for efficiency

### Research Directions

1. **Adaptive Prioritization**: Learn optimal priority function
2. **Curriculum Replay**: Sample based on learning progress
3. **Hindsight Experience Replay**: Generate synthetic experiences
4. **Episodic Memory**: Retrieve similar past experiences

---

## References

### Papers

1. **MuZero**: Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", Nature 2020
2. **Prioritized Experience Replay**: Schaul et al., "Prioritized Experience Replay", ICLR 2016
3. **Distributed PER**: Horgan et al., "Distributed Prioritized Experience Replay", ICLR 2018
4. **Reanalyze**: Schrittwieser et al., "Online and Offline Reinforcement Learning by Planning with a Learned Model", NeurIPS 2021

### Related Documentation

- [self_play_mcts.md](self_play_mcts.md) - Game history generation and MCTS
- [trainer.md](trainer.md) - Training loop and loss computation
- [neural_networks.md](neural_networks.md) - Model architecture and inference
- [shared_storage.md](shared_storage.md) - Coordination and weight management
- [main_orchestrator.md](main_orchestrator.md) - Overall system orchestration

### External Resources

- [Ray Documentation](https://docs.ray.io/) - Distributed computing framework
- [MuZero Pseudocode](https://arxiv.org/abs/1911.08265) - Algorithm details
- [OpenSpiel](https://github.com/deepmind/open_spiel) - Game environments

---

## Appendix

### Complete Configuration Example

```python
class ReplayBufferConfig:
    # Buffer settings
    replay_buffer_size = 5000
    batch_size = 256
    num_unroll_steps = 5
    td_steps = 10
    discount = 0.997
    
    # Observation settings
    stacked_observations = 4
    observation_shape = (3, 96, 96)
    action_space = list(range(18))
    
    # PER settings
    PER = True
    PER_alpha = 0.6
    
    # Reanalysis settings
    use_last_model_value = True
    reanalyse_on_gpu = True
    
    # Training settings
    training_steps = 100000
    seed = 42
```

### Glossary

- **Experience Replay**: Technique of storing and reusing past experiences for training
- **Prioritized Experience Replay (PER)**: Sampling experiences based on their learning value
- **TD-Error**: Temporal difference error, difference between predicted and target values
- **Importance Sampling**: Correction for bias introduced by non-uniform sampling
- **Reanalysis**: Re-evaluating past positions with updated model
- **Bootstrap Value**: Value estimate used to compute n-step returns
- **Absorbing State**: Terminal state with no further transitions
- **Gradient Scale**: Factor controlling gradient magnitude for different positions

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Maintainer**: MuZero Development Team

