"""Temporal state updater models for spacetime GNNs.

Provides RNN, LSTM, GRU, and Liquid Neural Network (LNN) wrappers for updating node states over time.
"""

import logging
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    from .liquid_neural_network import LiquidStateUpdater
    HAS_LNN = True
except ImportError:
    HAS_LNN = False
    LiquidStateUpdater = None

logger = logging.getLogger(__name__)


class RNNStateUpdater(nn.Module):
    """RNN-based state updater for temporal node states.
    
    Updates node states over time: h_i(t) = RNN(h_i(t-1), messages_received_at_t)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        """Initialize RNN state updater.
        
        Args:
            input_dim: Dimension of input messages
            hidden_dim: Hidden state dimension
            num_layers: Number of RNN layers
            dropout: Dropout rate (only for multi-layer RNNs)
            bidirectional: Whether to use bidirectional RNN
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for RNNStateUpdater")
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False
        )
        
        logger.info(
            f"Initialized RNNStateUpdater "
            f"(input_dim={input_dim}, hidden_dim={hidden_dim}, layers={num_layers})"
        )
    
    def forward(
        self,
        previous_state: torch.Tensor,
        messages: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update state given previous state and new messages.
        
        Args:
            previous_state: Previous node state [hidden_dim] or [batch, hidden_dim]
            messages: Aggregated messages from neighbors [message_dim] or [batch, message_dim]
            time_delta: Optional time delta since last update [1] or [batch, 1]
            
        Returns:
            Updated state [hidden_dim] or [batch, hidden_dim]
        """
        # Prepare input: concatenate previous state and messages
        if previous_state.dim() == 1:
            # Single node
            batch_size = 1
            previous_state = previous_state.unsqueeze(0)
            messages = messages.unsqueeze(0)
            if time_delta is not None:
                time_delta = time_delta.unsqueeze(0)
        else:
            batch_size = previous_state.shape[0]
        
        # Combine previous state and messages
        if messages.shape[-1] != self.input_dim:
            # Project messages to input_dim if needed
            if not hasattr(self, 'message_proj'):
                self.message_proj = nn.Linear(messages.shape[-1], self.input_dim).to(messages.device)
            messages = self.message_proj(messages)
        
        # Add time delta to input if provided
        if time_delta is not None:
            # Encode time delta and add to messages
            time_encoding = torch.sin(time_delta * 2 * torch.pi / 100.0)  # Simple encoding
            messages = messages + time_encoding
        
        # Combine state and messages
        # For RNN, we use messages as input and previous_state as initial hidden
        input_seq = messages.unsqueeze(0)  # [1, batch, input_dim]
        
        # Initialize hidden state
        hidden = previous_state.unsqueeze(0)  # [1, batch, hidden_dim]
        if self.num_layers > 1:
            # Expand for multiple layers
            hidden = hidden.repeat(self.num_layers, 1, 1)
        
        # Forward through RNN
        output, new_hidden = self.rnn(input_seq, hidden)
        
        # Return final hidden state
        new_state = new_hidden[-1]  # [batch, hidden_dim]
        
        if batch_size == 1:
            new_state = new_state.squeeze(0)
        
        return new_state
    
    def reset_state(self):
        """Reset internal state (if any)."""
        pass


class LSTMStateUpdater(nn.Module):
    """LSTM-based state updater with cell state for long-term memory.
    
    Handles vanishing gradient problem better than RNN.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        """Initialize LSTM state updater.
        
        Args:
            input_dim: Dimension of input messages
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate (only for multi-layer LSTMs)
            bidirectional: Whether to use bidirectional LSTM
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LSTMStateUpdater")
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False
        )
        
        logger.info(
            f"Initialized LSTMStateUpdater "
            f"(input_dim={input_dim}, hidden_dim={hidden_dim}, layers={num_layers})"
        )
    
    def forward(
        self,
        previous_state: torch.Tensor,
        messages: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update state given previous state and new messages.
        
        Args:
            previous_state: Previous node state [hidden_dim] or [batch, hidden_dim]
            messages: Aggregated messages from neighbors [message_dim] or [batch, message_dim]
            time_delta: Optional time delta since last update [1] or [batch, 1]
            cell_state: Optional previous cell state [hidden_dim] or [batch, hidden_dim]
            
        Returns:
            Tuple of (updated_state, updated_cell_state)
        """
        # Prepare input
        if previous_state.dim() == 1:
            batch_size = 1
            previous_state = previous_state.unsqueeze(0)
            messages = messages.unsqueeze(0)
            if time_delta is not None:
                time_delta = time_delta.unsqueeze(0)
            if cell_state is not None:
                cell_state = cell_state.unsqueeze(0)
        else:
            batch_size = previous_state.shape[0]
        
        # Project messages if needed
        if messages.shape[-1] != self.input_dim:
            if not hasattr(self, 'message_proj'):
                self.message_proj = nn.Linear(messages.shape[-1], self.input_dim).to(messages.device)
            messages = self.message_proj(messages)
        
        # Add time delta encoding
        if time_delta is not None:
            time_encoding = torch.sin(time_delta * 2 * torch.pi / 100.0)
            messages = messages + time_encoding
        
        # Prepare LSTM input
        input_seq = messages.unsqueeze(0)  # [1, batch, input_dim]
        
        # Initialize hidden and cell states
        hidden = previous_state.unsqueeze(0)  # [1, batch, hidden_dim]
        if cell_state is None:
            cell_state = torch.zeros_like(hidden)
        else:
            cell_state = cell_state.unsqueeze(0)
        
        if self.num_layers > 1:
            hidden = hidden.repeat(self.num_layers, 1, 1)
            cell_state = cell_state.repeat(self.num_layers, 1, 1)
        
        # Forward through LSTM
        output, (new_hidden, new_cell) = self.lstm(input_seq, (hidden, cell_state))
        
        # Return final states
        new_state = new_hidden[-1]  # [batch, hidden_dim]
        new_cell_state = new_cell[-1]  # [batch, hidden_dim]
        
        if batch_size == 1:
            new_state = new_state.squeeze(0)
            new_cell_state = new_cell_state.squeeze(0)
        
        return new_state, new_cell_state
    
    def reset_state(self):
        """Reset internal state."""
        pass


class GRUStateUpdater(nn.Module):
    """GRU-based state updater (faster, less memory than LSTM).
    
    Good balance for most use cases.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        """Initialize GRU state updater.
        
        Args:
            input_dim: Dimension of input messages
            hidden_dim: Hidden state dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate (only for multi-layer GRUs)
            bidirectional: Whether to use bidirectional GRU
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GRUStateUpdater")
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False
        )
        
        logger.info(
            f"Initialized GRUStateUpdater "
            f"(input_dim={input_dim}, hidden_dim={hidden_dim}, layers={num_layers})"
        )
    
    def forward(
        self,
        previous_state: torch.Tensor,
        messages: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update state given previous state and new messages.
        
        Args:
            previous_state: Previous node state [hidden_dim] or [batch, hidden_dim]
            messages: Aggregated messages from neighbors [message_dim] or [batch, message_dim]
            time_delta: Optional time delta since last update [1] or [batch, 1]
            
        Returns:
            Updated state [hidden_dim] or [batch, hidden_dim]
        """
        # Prepare input
        if previous_state.dim() == 1:
            batch_size = 1
            previous_state = previous_state.unsqueeze(0)
            messages = messages.unsqueeze(0)
            if time_delta is not None:
                time_delta = time_delta.unsqueeze(0)
        else:
            batch_size = previous_state.shape[0]
        
        # Project messages if needed
        if messages.shape[-1] != self.input_dim:
            if not hasattr(self, 'message_proj'):
                self.message_proj = nn.Linear(messages.shape[-1], self.input_dim).to(messages.device)
            messages = self.message_proj(messages)
        
        # Add time delta encoding
        if time_delta is not None:
            time_encoding = torch.sin(time_delta * 2 * torch.pi / 100.0)
            messages = messages + time_encoding
        
        # Prepare GRU input
        input_seq = messages.unsqueeze(0)  # [1, batch, input_dim]
        
        # Initialize hidden state
        hidden = previous_state.unsqueeze(0)  # [1, batch, hidden_dim]
        if self.num_layers > 1:
            hidden = hidden.repeat(self.num_layers, 1, 1)
        
        # Forward through GRU
        output, new_hidden = self.gru(input_seq, hidden)
        
        # Return final hidden state
        new_state = new_hidden[-1]  # [batch, hidden_dim]
        
        if batch_size == 1:
            new_state = new_state.squeeze(0)
        
        return new_state
    
    def reset_state(self):
        """Reset internal state."""
        pass

