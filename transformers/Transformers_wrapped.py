import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding instead of the sinusoidal one.
    """

    def __init__(self, max_seq_len, embed_dim):
        """
        Initialize the learned positional encoding.

        """
        super(LearnedPositionalEncoding, self).__init__()

        # Create a learnable position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Initialize the position embeddings
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        """
        Add positional encoding to the input.

        """
        seq_len = x.size(1)

        # Slice the positional embedding to the actual sequence length
        pos_emb = self.pos_embedding[:, :seq_len, :]

        # Add positional encoding to the input (broadcast across batch dimension)
        x = x + pos_emb

        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.

    standard transformer encoder block with:
    - Multi-head self-attention
    - Feed-forward network
    - LayerNorm and residual connections
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Initialize the transformer block.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward network hidden dimension.
            dropout (float): Dropout rate.
        """
        super(TransformerBlock, self).__init__()

        # Check that embedding dimension is divisible by the number of heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} must be divisible by number of heads {num_heads}")

        # Calculate attention head dimension
        self.head_dim = embed_dim // num_heads

        # Multi-head self-attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Dropout after attention
        self.attention_dropout = nn.Dropout(dropout)

        # Layer normalization for attention
        self.attention_norm = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff_linear1 = nn.Linear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.ff_linear2 = nn.Linear(ff_dim, embed_dim)

        # Dropout for feed-forward network
        self.ffn_dropout = nn.Dropout(dropout)

        # Layer normalization for feed-forward
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Process input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Self-attention with residual connection
        # First normalize (pre-norm architecture)
        norm_x = self.attention_norm(x)

        # Self-attention using PyTorch's MultiheadAttention
        attn_output, _ = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x
        )

        # Apply dropout and residual connection
        attn_output = self.attention_dropout(attn_output)
        x = x + attn_output

        # Feed-forward network with residual connection
        # Normalize again (pre-norm architecture)
        norm_x = self.ffn_norm(x)

        # Two-layer feed-forward network
        ff_output = self.ff_linear1(norm_x)
        ff_output = self.relu(ff_output)
        ff_output = self.ff_linear2(ff_output)

        # Apply dropout and residual connection
        ff_output = self.ffn_dropout(ff_output)
        x = x + ff_output

        return x


class TransformerModel(nn.Module):
    """
    Transformer model for sequence classification.

    Architecture:
    - Convolutional embedding layer
    - Positional encoding
    - Series of transformer blocks
    - Global pooling
    - Fully connected layer
    - Classification layer
    """

    def __init__(self, input_shape, num_classes, hp_dict):
        """
        Initialize the transformer model.

        Args:
            input_shape (tuple): (time_steps, features)
            num_classes (int): Number of output classes.
            hp_dict (dict): Dictionary of hyperparameters.
        """
        super(TransformerModel, self).__init__()

        # Extract hyperparameters with defaults
        time_steps, input_dim = input_shape
        self.num_classes = num_classes
        # Embedding dimension
        self.embed_dim = hp_dict.get('embed_dim', 128)

        # Convolutional embedding layer
        kernel_size = hp_dict.get('kernel_size', 3)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.embed_dim,
            kernel_size=kernel_size,
            padding='same'
        )

        # Activation for conv
        self.relu = nn.ReLU()

        # Positional encoding
        self.pos_encoding = LearnedPositionalEncoding(
            max_seq_len=time_steps,
            embed_dim=self.embed_dim
        )

        # Number of transformer blocks
        self.num_blocks = hp_dict.get('num_blocks', 3)

        # Create transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=hp_dict.get('num_heads', 4),
                ff_dim=hp_dict.get('ff_dim', 256),
                dropout=hp_dict.get('block_dropout', 0.1)
            )
            for _ in range(self.num_blocks)
        ])

        # Global pool options: 'mean', 'max', 'first'
        pool_type = hp_dict.get('pool_type', 'mean')
        if pool_type == 'mean':
            self.global_pool = lambda x: torch.mean(x, dim=1)
        elif pool_type == 'max':
            self.global_pool = lambda x: torch.max(x, dim=1)[0]
        elif pool_type == 'first':
            self.global_pool = lambda x: x[:, 0, :]
        else:
            raise ValueError(f"Invalid pool_type: {pool_type}")

        # Fully connected layer
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)

        # Dropout for head
        head_dropout_rate = hp_dict.get('head_dropout', 0.2)
        self.head_dropout = nn.Dropout(head_dropout_rate)

        # Classification layer
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, features).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Permute for Conv1d: [batch, time_steps, channels] -> [batch, channels, time_steps]
        x = x.permute(0, 2, 1)

        # Apply convolutional embedding
        x = self.conv(x)
        x = self.relu(x)

        # Permute back to [batch, time_steps, embed_dim]
        x = x.permute(0, 2, 1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x)

        # Apply dense layer
        x = self.dense(x)
        x = self.relu(x)
        x = self.head_dropout(x)

        # Final classification
        logits = self.classifier(x)

        return logits

    def get_probabilities(self, x):
        """
        Get probability outputs by applying softmax to the model output.

        Args:
            x (torch.Tensor): Input tensor or logits.

        Returns:
            torch.Tensor: Probability outputs.
        """
        logits = x if x.size(-1) == self.classifier.out_features else self.forward(x)
        return F.softmax(logits, dim=-1)


class TransformerModelBuilder:
    """Handles building Transformer models with hyperparameter tuning."""

    @staticmethod
    def build_model_hp(hp, input_shape, num_classes):
        """
        Build a Transformer model with hyperparameters.

        Args:
            hp: Hyperparameter object.
            input_shape (tuple): (time_steps, features)
            num_classes (int): Number of output classes.

        Returns:
            tuple: (model, learning_rate)
        """

        try:
            hp_dict = {
                'embed_dim': hp.Choice('embed_dim', values=[64, 128, 256]),
                'kernel_size': hp.Choice('kernel_size', values=[3, 5, 7]),
                'num_blocks': hp.Int('num_blocks', min_value=2, max_value=6, step=1),
                'num_heads': hp.Choice('num_heads', values=[2, 4, 8]),
                'ff_dim': hp.Int('ff_dim', min_value=128, max_value=512, step=128),
                'block_dropout': hp.Float('block_dropout', min_value=0.0, max_value=0.3, step=0.1),
                'pool_type': hp.Choice('pool_type', values=['mean', 'max', 'first']),
                'head_dropout': hp.Float('head_dropout', min_value=0.0, max_value=0.5, step=0.1),
            }

            # Learning rate
            learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')

            # Create the model
            model = TransformerModel(input_shape, num_classes, hp_dict)
            return model, learning_rate

        except Exception as e:
            logger.error(f"Error during Transformer model building: {e}")
            return None, None

    @staticmethod
    def build_transformer_model_final(hp_dict, input_shape, num_classes):
        """
        Builds the final Transformer model using the best hyperparameters found.

        Args:
            hp_dict (dict): Dictionary with hyperparameter values.
            input_shape (tuple): (time_steps, features)
            num_classes (int): Number of output classes.

        Returns:
            TransformerModel: A PyTorch model instance.
        """
        try:
            model = TransformerModel(input_shape, num_classes, hp_dict)
            return model
        except Exception as e:
            logger.error(f"Error during final Transformer model building: {e}")
            return None