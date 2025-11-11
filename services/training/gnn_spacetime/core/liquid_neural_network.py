"""Liquid Neural Network (LNN) for continuous-time temporal modeling.

LNNs provide superior spacetime representations with:
- Continuous-time state evolution
- Temporal robustness to novel environments
- Efficiency and interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class LiquidLayer(nn.Module):
    """Liquid Neural Network layer for continuous-time dynamics.
    
    Implements a simplified LNN that processes temporal information
    with leaky integration and adaptive time constants.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        time_constant: float = 1.0,
        dt: float = 0.01
    ):
        """Initialize liquid layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension
            time_constant: Time constant for temporal dynamics (tau)
            dt: Time step for continuous-time integration
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_constant = time_constant
        self.dt = dt
        
        # Input-to-hidden transformation
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        
        # Hidden-to-hidden (recurrent) connections
        self.hidden_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Hidden-to-output projection
        self.output_transform = nn.Linear(hidden_dim, output_dim)
        
        # Adaptive time constant (learnable)
        self.tau = nn.Parameter(torch.tensor(time_constant))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.hidden_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        
        if self.input_transform.bias is not None:
            nn.init.zeros_(self.input_transform.bias)
        if self.output_transform.bias is not None:
            nn.init.zeros_(self.output_transform.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with continuous-time dynamics.
        
        Implements: dh/dt = -h/τ + f(x, h)
        
        Args:
            x: Input tensor [batch_size, input_dim]
            h_prev: Previous hidden state [batch_size, hidden_dim]
            time_delta: Time delta for integration [batch_size] or scalar
            
        Returns:
            output: Output tensor [batch_size, output_dim]
            h_new: New hidden state [batch_size, hidden_dim]
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Get time delta (use provided or default)
        if time_delta is None:
            dt = torch.tensor(self.dt, device=x.device)
        else:
            dt = time_delta if isinstance(time_delta, torch.Tensor) else torch.tensor(time_delta, device=x.device)
            if dt.dim() == 0:
                dt = dt.expand(batch_size)
        
        # Transform input
        input_activation = self.input_transform(x)  # [batch, hidden_dim]
        
        # Recurrent activation
        hidden_activation = self.hidden_transform(h_prev)  # [batch, hidden_dim]
        
        # Combined activation
        f_h = torch.tanh(input_activation + hidden_activation)  # f(x, h)
        
        # Continuous-time dynamics: dh/dt = -h/τ + f(x, h)
        # Using Euler integration: h_new = h_old + dt * (-h_old/τ + f(x, h_old))
        tau = torch.clamp(self.tau, min=0.01)  # Ensure positive
        decay_term = -h_prev / tau.unsqueeze(0) if tau.dim() == 0 else -h_prev / tau
        dh_dt = decay_term + f_h
        
        # Integrate over time
        if dt.dim() == 1:
            dt = dt.unsqueeze(1)  # [batch, 1]
        h_new = h_prev + dt * dh_dt
        
        # Project to output
        output = self.output_transform(h_new)
        
        return output, h_new


class LiquidStateUpdater(nn.Module):
    """Liquid Neural Network state updater for temporal nodes.
    
    Replaces RNN/LSTM/GRU with LNN for more robust temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        time_constant: float = 1.0,
        dt: float = 0.01
    ):
        """Initialize liquid state updater.
        
        Args:
            input_dim: Input dimension (messages + previous state)
            hidden_dim: Hidden state dimension
            time_constant: Time constant for temporal dynamics
            dt: Time step for integration
        """
        super().__init__()
        self.liquid_layer = LiquidLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            time_constant=time_constant,
            dt=dt
        )
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        messages: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update node state using liquid dynamics.
        
        Args:
            messages: Aggregated messages from neighbors [batch_size, message_dim]
            prev_state: Previous hidden state [batch_size, hidden_dim]
            time_delta: Time delta for integration
            
        Returns:
            Updated hidden state [batch_size, hidden_dim]
        """
        # If prev_state is provided, concatenate with messages
        if prev_state is not None:
            combined_input = torch.cat([messages, prev_state], dim=-1)
        else:
            combined_input = messages
        
        # Ensure input matches expected dimension
        if combined_input.size(-1) != self.liquid_layer.input_dim:
            # Project to correct dimension
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(combined_input.size(-1), self.liquid_layer.input_dim).to(combined_input.device)
            combined_input = self.input_proj(combined_input)
        
        # Forward through liquid layer
        _, h_new = self.liquid_layer(combined_input, prev_state, time_delta)
        
        return h_new


class LiquidEdgeWeightUpdater(nn.Module):
    """Liquid Neural Network for time-varying edge weights.
    
    Models edge weight evolution w(t) using continuous-time dynamics.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        time_constant: float = 1.0
    ):
        """Initialize liquid edge weight updater.
        
        Args:
            input_dim: Input dimension (node features + relation embedding)
            hidden_dim: Hidden state dimension
            time_constant: Time constant for weight evolution
        """
        super().__init__()
        self.liquid_layer = LiquidLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,  # Single weight value
            time_constant=time_constant
        )
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        relation_embedding: torch.Tensor,
        prev_weight: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update edge weight using liquid dynamics.
        
        Args:
            source_features: Source node features [batch, node_dim]
            target_features: Target node features [batch, node_dim]
            relation_embedding: Relation embedding [batch, rel_dim]
            prev_weight: Previous weight value [batch, 1]
            time_delta: Time delta for integration
            
        Returns:
            Updated weight [batch, 1]
        """
        # Combine inputs
        combined = torch.cat([source_features, target_features, relation_embedding], dim=-1)
        
        # If prev_weight provided, use as hidden state
        h_prev = prev_weight if prev_weight is not None else None
        
        # Forward through liquid layer
        weight, _ = self.liquid_layer(combined, h_prev, time_delta)
        
        # Ensure weight is in valid range [0, 1] or [-1, 1]
        weight = torch.sigmoid(weight)  # [0, 1]
        
        return weight

