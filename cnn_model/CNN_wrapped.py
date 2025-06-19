import torch
import torch.nn as nn
import logging
from Gait_Phase.gpu_utils import configure_gpu

configure_gpu()


#########################################
# CNN Model Definition
#########################################
class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes, hp_dict):
        """
        Builds a CNN model with an explicit "Input" layer equivalent.

        Architecture:
          - (Simulated) Input layer.
          - A variable number of Conv1D layers with hyperparameter-defined filters, kernel sizes, activations, and optional dropout.
          - Global max pooling.
          - Dense (fully connected) layer followed by dropout.
          - Output layer with softmax activation.

        Args:
            input_shape (tuple): (time_steps, features)
            num_classes (int): Number of output classes.
            hp_dict (dict): Dictionary of hyperparameters.
        """
        super(CNNModel, self).__init__()
        self.input_time_steps, in_channels = input_shape
        self.num_classes = num_classes
        conv_layers = hp_dict.get('conv_layers', 1)
        self.conv_blocks = nn.ModuleList()
        current_channels = in_channels

        for i in range(conv_layers):
            # Get hyperparameters.
            filters = hp_dict.get(f'filters_{i}', 32)
            kernel_size = hp_dict.get(f'kernel_size_{i}', 3)
            activation_choice = hp_dict.get(f'activation_{i}', 'relu')
            # Compute padding for "same" output dimensions.
            padding = kernel_size // 2

            # Convolutional layer.
            conv = nn.Conv1d(in_channels=current_channels,
                             out_channels=filters,
                             kernel_size=kernel_size,
                             padding=padding)
            self.conv_blocks.append(conv)
            # Activation layer.
            if activation_choice == 'relu':
                self.conv_blocks.append(nn.ReLU())
            elif activation_choice == 'tanh':
                self.conv_blocks.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation_choice}")

            # Optional dropout.
            if hp_dict.get(f'dropout_{i}', False):
                dropout_rate = hp_dict.get(f'dropout_rate_{i}', 0.1)
                self.conv_blocks.append(nn.Dropout(p=dropout_rate))

            current_channels = filters

        # Global max pooling over the time dimension.
        self.global_pool = nn.AdaptiveMaxPool1d(output_size=1)

        # Dense layers.
        dense_units = hp_dict.get('dense_units', 64)
        dense_dropout = hp_dict.get('dense_dropout', 0.1)
        self.dense = nn.Linear(current_channels, dense_units)
        self.dense_activation = nn.ReLU()
        self.dense_dropout = nn.Dropout(p=dense_dropout)
        self.classifier = nn.Linear(dense_units, num_classes)
        # Keep softmax for predictions
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time_steps, features).

        Returns:
            torch.Tensor: Output logits of shape (batch, num_classes).

        Note: We also provide a method get_probabilities() to apply softmax when needed.
        """
        # Permute to (batch, channels, time_steps)
        x = x.permute(0, 2, 1)
        # Process through convolutional blocks.
        for layer in self.conv_blocks:
            x = layer(x)
        # Global max pooling.
        x = self.global_pool(x)  # shape becomes (batch, channels, 1)
        x = x.squeeze(-1)  # shape: (batch, channels)
        # Dense layers.
        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.dense_dropout(x)
        x = self.classifier(x)
        return x

    def get_probabilities(self, x):
        """
        Get probability outputs by applying softmax to the model output.

        Args:
            x (torch.Tensor): Input tensor or logits.

        Returns:
            torch.Tensor: Probability outputs.
        """
        logits = x if x.size(-1) == self.classifier.out_features else self.forward(x)
        return self.softmax(logits)


#########################################
# CNN Model Builder
#########################################
class CNNModelBuilder:
    """Handles building CNN models with hyperparameter tuning using an explicit Input layer."""

    @staticmethod
    def build_model_hp(hp, input_shape, num_classes):
        """
        Build a model with hyperparameters.

        Args:
            hp: Hyperparameter object.
            input_shape (tuple): (time_steps, features)
            num_classes (int): Number of output classes.

        Returns:
            tuple: (model, learning_rate)
        """
        try:
            hp_dict = {}
            # Number of convolutional layers.
            conv_layers = hp.Int('conv_layers', min_value=1, max_value=3, step=1)
            hp_dict['conv_layers'] = conv_layers

            for i in range(conv_layers):
                hp_dict[f'filters_{i}'] = hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32)
                hp_dict[f'kernel_size_{i}'] = hp.Choice(f'kernel_size_{i}', values=[3, 5, 7])
                hp_dict[f'activation_{i}'] = hp.Choice(f'activation_{i}', values=['relu', 'tanh'])
                if hp.Boolean(f'dropout_{i}', default=False):
                    hp_dict[f'dropout_{i}'] = True
                    hp_dict[f'dropout_rate_{i}'] = hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1,
                                                            default=0.1)
                else:
                    hp_dict[f'dropout_{i}'] = False

            # Dense layers.
            hp_dict['dense_units'] = hp.Int('dense_units', min_value=64, max_value=256, step=64, default=64)
            hp_dict['dense_dropout'] = hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1, default=0.1)
            model = CNNModel(input_shape, num_classes, hp_dict)
            # Return a tuple with a default learning rate
            return model, 1e-3
        except Exception as e:
            logging.getLogger(__name__).error(f"Error during CNN model building: {e}")
            return None, None

    @staticmethod
    def build_cnn_model_final(hp_dict, input_shape, num_classes):
        """
        Builds the final CNN model using the best hyperparameters found.
        Uses an explicit input layer for getting correct tensor dimensions.

        Args:
            hp_dict (dict): Dictionary with hyperparameter values.
            input_shape (tuple): (time_steps, features)
            num_classes (int): Number of output classes.

        Returns:
            CNNModel: A PyTorch model instance.
        """
        try:
            model = CNNModel(input_shape, num_classes, hp_dict)
            return model
        except Exception as e:
            logging.getLogger(__name__).error(f"Error during final CNN model building: {e}")
            return None

