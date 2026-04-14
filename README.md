# Reinforcement Learning Models of Avoidance Learning in Anxiety

### Overview

This pipeline fits reinforcement learning models to behavioural data from a probabilistic reversal-learning task, compares parameter estimates between high- and low-anxiety groups, and validates the modelling pipeline through parameter and model recovery.

### Research Question

Do anxious individuals learn differently when trying to avoid aversive outcomes?

### Pipeline

1. **Simulate** behaviour from RL models (generative modelling)
2. **Write likelihood functions** and fit models via maximum likelihood estimation
3. **Compare** models using information criteria (AIC, BIC)
4. **Validate** the pipeline with parameter recovery and model recovery
5. **Draw inferences** about group differences in computational mechanisms

### Models

| Model | Free Parameters | Update Rule |
|:---|:---|:---|
| **Model 1** | α (learning rate), β (inverse temperature) | $V_i^{(t+1)} = V_i^{(t)} + \alpha(o^{(t)} - V_i^{(t)})$ |
| **Model 2** | α, β, A (memory decay) | $V_i^{(t+1)} = A \cdot V_i^{(t)} + \alpha(o^{(t)} - V_i^{(t)})$ |
| **Model 3** | α⁺ (neutral LR), α⁻ (punishment LR), β | $V_i^{(t+1)} = V_i^{(t)} + [(1-o^{(t)})\alpha^+ + o^{(t)}\alpha^-](o^{(t)} - V_i^{(t)})$ |

All three models use the same **softmax** action selection with negative β (avoidance):

$$P(\text{choose A}) = \frac{\exp(-\beta \cdot V_A)}{\exp(-\beta \cdot V_A) + \exp(-\beta \cdot V_B)}$$
