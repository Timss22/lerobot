# SmolVLA Study Guide: How to Learn and Modify the Code

This guide will help you understand the core of SmolVLA in LeRobot and show you exactly where to make modifications.

## 📚 Repository Structure Overview

The SmolVLA implementation is located in:
```
src/lerobot/policies/smolvla/
├── configuration_smolvla.py    # Configuration (learning rate, batch size, architecture params)
├── modeling_smolvla.py        # Main model architecture (VLAFlowMatching, SmolVLAPolicy)
├── smolvlm_with_expert.py    # VLM + Expert architecture implementation
└── processor_smolvla.py       # Input/output preprocessing
```

## 🎯 Key Files to Study

### 1. **Configuration File** (`configuration_smolvla.py`)
**Location:** `src/lerobot/policies/smolvla/configuration_smolvla.py`

This is where you configure:
- **Learning rate and optimizer settings** (lines 76-84)
- **Training hyperparameters** (batch size is in `src/lerobot/configs/train.py`)
- **Architecture parameters** (VLM layers, expert layers, attention modes)
- **Input/output dimensions** (state/action dimensions, image sizes)

### 2. **Model Architecture** (`modeling_smolvla.py`)
**Location:** `src/lerobot/policies/smolvla/modeling_smolvla.py`

Contains:
- `SmolVLAPolicy`: Main policy wrapper (line 216)
- `VLAFlowMatching`: Core model architecture (line 448)
- Forward pass logic
- Action prediction logic

### 3. **VLM + Expert Architecture** (`smolvlm_with_expert.py`)
**Location:** `src/lerobot/policies/smolvla/smolvlm_with_expert.py`

Contains:
- `SmolVLMWithExpertModel`: The actual neural network architecture (line 61)
- Vision encoder, language model, and action expert integration
- Attention mechanisms (self-attention vs cross-attention)

### 4. **Input/Output Processing** (`processor_smolvla.py`)
**Location:** `src/lerobot/policies/smolvla/processor_smolvla.py`

Handles:
- Image preprocessing
- State normalization
- Language tokenization
- Action denormalization

## 🔧 Where to Modify Key Parameters

### **Learning Rate and Optimizer Settings**

**File:** `src/lerobot/policies/smolvla/configuration_smolvla.py`

**Lines 76-84:**
```python
optimizer_lr: float = 1e-4                    # ← Change learning rate here
optimizer_betas: tuple[float, float] = (0.9, 0.95)
optimizer_eps: float = 1e-8
optimizer_weight_decay: float = 1e-10
optimizer_grad_clip_norm: float = 10

scheduler_warmup_steps: int = 1_000            # ← Warmup steps
scheduler_decay_steps: int = 30_000           # ← Decay steps
scheduler_decay_lr: float = 2.5e-6            # ← Final learning rate after decay
```

**How to override via command line:**
```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.optimizer_lr=2e-4 \
  --policy.scheduler_warmup_steps=2000 \
  --dataset.repo_id=your_dataset
```

### **Batch Size**

**File:** `src/lerobot/configs/train.py`

**Line 55:**
```python
batch_size: int = 8  # ← Default batch size
```

**How to override via command line:**
```bash
lerobot-train \
  --policy.type=smolvla \
  --batch_size=64 \
  --dataset.repo_id=your_dataset
```

### **Input/Output Dimensions**

**File:** `src/lerobot/policies/smolvla/configuration_smolvla.py`

**Lines 30-44:**
```python
n_obs_steps: int = 1              # Number of observation steps
chunk_size: int = 50              # Action chunk size
n_action_steps: int = 50         # Number of action steps to predict

max_state_dim: int = 32          # ← Maximum state dimension (padded if smaller)
max_action_dim: int = 32         # ← Maximum action dimension (padded if smaller)

resize_imgs_with_padding: tuple[int, int] = (512, 512)  # ← Image size
```

### **Architecture Modifications**

#### **1. VLM Backbone Selection**

**File:** `src/lerobot/policies/smolvla/configuration_smolvla.py`

**Line 86:**
```python
vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # ← Change VLM model
```

#### **2. Number of Layers**

**File:** `src/lerobot/policies/smolvla/configuration_smolvla.py`

**Lines 97-100:**
```python
num_expert_layers: int = -1              # ← Expert layers (-1 = same as VLM)
num_vlm_layers: int = 16                 # ← Number of VLM layers to use
self_attn_every_n_layers: int = 2        # ← Self-attention frequency
expert_width_multiplier: float = 0.75     # ← Expert hidden size multiplier
```

#### **3. Attention Mode**

**File:** `src/lerobot/policies/smolvla/configuration_smolvla.py`

**Line 91:**
```python
attention_mode: str = "cross_attn"  # Options: "cross_attn" or "self_attn"
```

#### **4. Freezing Components**

**File:** `src/lerobot/policies/smolvla/configuration_smolvla.py`

**Lines 71-73:**
```python
freeze_vision_encoder: bool = True    # ← Freeze vision encoder
train_expert_only: bool = True        # ← Only train expert (not VLM)
train_state_proj: bool = True        # ← Train state projection layer
```

#### **5. Modify Architecture Components**

**File:** `src/lerobot/policies/smolvla/modeling_smolvla.py`

**Lines 490-501:** Projection layers
```python
self.state_proj = nn.Linear(...)        # ← State projection
self.action_in_proj = nn.Linear(...)   # ← Action input projection
self.action_out_proj = nn.Linear(...)  # ← Action output projection
```

**File:** `src/lerobot/policies/smolvla/smolvlm_with_expert.py`

**Lines 61-134:** Core architecture
- Modify `SmolVLMWithExpertModel.__init__()` to change architecture
- Modify `forward()` method (line 404) to change forward pass
- Modify attention mechanisms in `forward_attn_layer()` and `forward_cross_attn_layer()`

### **Input/Output Processing**

**File:** `src/lerobot/policies/smolvla/processor_smolvla.py`

**Lines 39-103:** Pre/post-processing pipeline
- Modify `make_smolvla_pre_post_processors()` to change preprocessing
- Add custom processor steps in the pipeline

## 🚀 How to Train with Custom Settings

### Example 1: Change Learning Rate and Batch Size

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.optimizer_lr=5e-4 \
  --batch_size=32 \
  --dataset.repo_id=your_dataset \
  --steps=100000
```

### Example 2: Modify Architecture

Create a custom config file or override via CLI:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.num_vlm_layers=12 \
  --policy.expert_width_multiplier=0.5 \
  --policy.attention_mode=self_attn \
  --dataset.repo_id=your_dataset
```

### Example 3: Train from Scratch (No Pretrained Weights)

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=False \
  --policy.train_expert_only=False \
  --dataset.repo_id=your_dataset
```

## 📖 Understanding the Architecture Flow

1. **Input Processing** (`processor_smolvla.py`):
   - Images → Vision encoder embeddings
   - State → State projection → Language embeddings
   - Language task → Tokenized embeddings

2. **Forward Pass** (`modeling_smolvla.py`, `smolvlm_with_expert.py`):
   - Vision encoder processes images
   - VLM processes vision + language tokens
   - Expert processes action tokens with cross-attention to VLM
   - Flow matching predicts action sequence

3. **Output Processing** (`processor_smolvla.py`):
   - Denormalize actions
   - Return action chunk

## 🔍 Key Functions to Study

### In `modeling_smolvla.py`:
- `SmolVLAPolicy._get_action_chunk()` (line 248): How actions are generated
- `VLAFlowMatching.forward()` (line 448): Main forward pass
- `VLAFlowMatching._forward_flow_matching()`: Flow matching implementation

### In `smolvlm_with_expert.py`:
- `SmolVLMWithExpertModel.forward()` (line 404): Core architecture forward
- `forward_attn_layer()` (line 198): Self-attention mechanism
- `forward_cross_attn_layer()` (line 275): Cross-attention mechanism

## 📝 Training Script Location

**File:** `src/lerobot/scripts/lerobot_train.py`

This is the main training script. It:
- Loads configuration
- Creates dataloader
- Sets up optimizer/scheduler
- Runs training loop

## 🎓 Recommended Learning Path

1. **Start with Configuration** (`configuration_smolvla.py`):
   - Understand all parameters
   - Try changing learning rate and batch size
   - Experiment with architecture parameters

2. **Study the Architecture** (`smolvlm_with_expert.py`):
   - Understand how VLM and expert interact
   - Study attention mechanisms
   - See how vision, language, and actions are combined

3. **Understand Forward Pass** (`modeling_smolvla.py`):
   - Trace through `VLAFlowMatching.forward()`
   - Understand flow matching
   - See how actions are predicted

4. **Modify Input/Output** (`processor_smolvla.py`):
   - Understand preprocessing pipeline
   - Modify normalization
   - Add custom processing steps

5. **Run Experiments**:
   - Start with small changes (learning rate, batch size)
   - Progress to architecture modifications
   - Use the training script to test changes

## 🛠️ Quick Reference: Common Modifications

| What to Change | File | Line(s) | Parameter |
|---------------|------|---------|-----------|
| Learning Rate | `configuration_smolvla.py` | 76 | `optimizer_lr` |
| Batch Size | `configs/train.py` | 55 | `batch_size` |
| State Dimension | `configuration_smolvla.py` | 43 | `max_state_dim` |
| Action Dimension | `configuration_smolvla.py` | 44 | `max_action_dim` |
| Image Size | `configuration_smolvla.py` | 47 | `resize_imgs_with_padding` |
| VLM Layers | `configuration_smolvla.py` | 98 | `num_vlm_layers` |
| Expert Layers | `configuration_smolvla.py` | 97 | `num_expert_layers` |
| Expert Width | `configuration_smolvla.py` | 100 | `expert_width_multiplier` |
| Attention Mode | `configuration_smolvla.py` | 91 | `attention_mode` |
| VLM Model | `configuration_smolvla.py` | 86 | `vlm_model_name` |

## 📚 Additional Resources

- **Documentation:** `docs/source/policy_smolvla_README.md`
- **Example Usage:** `examples/tutorial/smolvla/using_smolvla_example.py`
- **Training Example:** `examples/training/train_policy.py`
- **Paper:** https://arxiv.org/abs/2506.01844

## 💡 Tips for Experimentation

1. **Start Small**: Change one parameter at a time (e.g., just learning rate)
2. **Use CLI Overrides**: Test changes without modifying code
3. **Check Logs**: Monitor training metrics to see impact of changes
4. **Version Control**: Commit before making major changes
5. **Read Error Messages**: They often point to the exact line causing issues

Happy learning and experimenting! 🚀

