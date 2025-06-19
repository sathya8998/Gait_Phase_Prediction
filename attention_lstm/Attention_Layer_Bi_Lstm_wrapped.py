import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """
    Attention mechanism to focus on important features in a sequence.

    """

    def __init__(self, hidden_size):
        """
        Initialize the attention layer.

        Args:
            hidden_size (int): Size of the hidden vectors.
        """
        super(AttentionLayer, self).__init__()

        self.initialized = False
        self.attention_weights = None  # Store for analysis

        # Query, Key, Value projections for attention
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

    def _initialize(self, batch_size, time_steps, hidden_size):
        """
        Args:
            batch_size (int): Size of the batch.
            time_steps (int): Length of the sequence.
            hidden_size (int): Size of the hidden vectors.
        """
        if self.initialized:
            return

        # Set initialized flag
        self.initialized = True

        # Log initialization for debugging
        logger.debug(f"Initializing attention layer with dimensions: "
                     f"batch_size={batch_size}, time_steps={time_steps}, hidden_size={hidden_size}")


    def forward(self, x):
        """
        Apply attention mechanism to the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, hidden_size).

        Returns:
            torch.Tensor: Context vector of shape (batch_size, hidden_size).
        """
        batch_size, time_steps, hidden_size = x.size()

        # Initialize weights if needed
        self._initialize(batch_size, time_steps, hidden_size)

        # Project input to query, key, value
        query = self.W_q(x)  # (batch_size, time_steps, hidden_size)
        key = self.W_k(x)  # (batch_size, time_steps, hidden_size)
        value = self.W_v(x)  # (batch_size, time_steps, hidden_size)

        # Calculate attention scores
        # Dot product between query and key for each time step
        scores = torch.bmm(query, key.transpose(1, 2))  # (batch_size, time_steps, time_steps)

        # Scale scores by sqrt(hidden_size)
        scores = scores / (hidden_size ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, time_steps, time_steps)

        # Store weights for later analysis
        self.attention_weights = attention_weights.detach()

        # Apply attention weights to values
        context = torch.bmm(attention_weights, value)  # (batch_size, time_steps, hidden_size)

        # Average along the time dimension to get a context vector
        context = torch.mean(context, dim=1)  # (batch_size, hidden_size)

        return context

    def get_attention_weights(self):
        """Get the last computed attention weights for analysis."""
        return self.attention_weights


class LSTMModel(nn.Module):
    def __init__(self, input_shape, num_classes, hp_dict):
        super(LSTMModel, self).__init__()

        # Extract and store hyperparameters
        self.time_steps, input_dim = input_shape
        self.num_classes = num_classes
        # LSTM units
        self.units_1 = hp_dict.get('units_1', 32)
        bidirectional = hp_dict.get('bidirectional', True)
        self.units_2 = hp_dict.get('units_2', 32)
        self.fc_units = hp_dict.get('fc_units', 32)

        # Calculate actual output dimensions considering bidirectionality
        lstm_output_dim = self.units_1 * 2 if bidirectional else self.units_1
        second_output_dim = self.units_2 * 2 if bidirectional else self.units_2

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.units_1,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Batch normalization after first LSTM
        # used lstm_output_dim
        self.bn1 = nn.BatchNorm1d(lstm_output_dim)

        # Dropout after first LSTM
        dropout_1 = hp_dict.get('dropout_1', 0.2)
        self.dropout1 = nn.Dropout(dropout_1)

        # Second LSTM layer (optional)
        self.second_lstm = hp_dict.get('second_lstm', False)
        if self.second_lstm:
            self.lstm2 = nn.LSTM(
                input_size=lstm_output_dim,
                hidden_size=self.units_2,
                batch_first=True,
                bidirectional=bidirectional
            )

            # Batch normalization after second LSTM
            # used second_output_dim
            self.bn2 = nn.BatchNorm1d(second_output_dim)

            # Dropout after second LSTM
            dropout_2 = hp_dict.get('dropout_2', 0.2)
            self.dropout2 = nn.Dropout(dropout_2)

            # Input dimension for attention is from second LSTM
            attention_input_dim = second_output_dim
        else:
            # Input dimension for attention is from first LSTM
            attention_input_dim = lstm_output_dim

        # Attention layer
        self.attention = AttentionLayer(attention_input_dim)

        # Fully connected layer
        self.fc = nn.Linear(attention_input_dim, self.fc_units)
        self.fc_activation = nn.ReLU()

        # Batch normalization after FC
        self.bn_fc = nn.BatchNorm1d(self.fc_units)

        # Dropout after FC
        fc_dropout_rate = hp_dict.get('fc_dropout', 0.5)
        self.fc_dropout = nn.Dropout(fc_dropout_rate)

        # Output layer
        self.classifier = nn.Linear(self.fc_units, num_classes)

        # Softmax activation (for probability outputs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        lstm1_out, _ = self.lstm1(x)

        # Permute to (batch, channels, time_steps) for batch norm
        lstm1_out = lstm1_out.permute(0, 2, 1)
        lstm1_out = self.bn1(lstm1_out)
        lstm1_out = lstm1_out.permute(0, 2, 1)
        lstm1_out = self.dropout1(lstm1_out)

        if self.second_lstm:
            lstm2_out, _ = self.lstm2(lstm1_out)
            lstm2_out = lstm2_out.permute(0, 2, 1)
            lstm2_out = self.bn2(lstm2_out)
            lstm2_out = lstm2_out.permute(0, 2, 1)
            lstm2_out = self.dropout2(lstm2_out)
            attention_input = lstm2_out
        else:
            attention_input = lstm1_out

        context = self.attention(attention_input)
        fc_out = self.fc(context)
        fc_out = self.fc_activation(fc_out)
        fc_out = self.bn_fc(fc_out)
        fc_out = self.fc_dropout(fc_out)
        logits = self.classifier(fc_out)
        return logits

    def get_probabilities(self, x):
        logits = x if x.size(-1) == self.classifier.out_features else self.forward(x)
        return self.softmax(logits)


class LSTMModelBuilder:
    """Handles building LSTM models with hyperparameter tuning."""

    @staticmethod
    def build_model_hp(hp, input_shape, num_classes):
        """
        Build an LSTM model with hyperparameters.

        Args:
            hp: Hyperparameter object.
            input_shape (tuple): (time_steps, features)
            num_classes (int): Number of output classes.

        Returns:
            tuple: (model, learning_rate)
        """
        try:
            hp_dict = {
                 #LSTM hyperparameters: use small network values
                 'units_1': hp.Int('units_1', min_value=32, max_value=128, step=32),
                 'bidirectional': hp.Boolean('bidirectional', default=True),
                 'dropout_1': hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1),
                 # Second LSTM layer (optional)
                 'second_lstm': hp.Boolean('second_lstm', default=True),
                 'units_2': hp.Int('units_2', min_value=16, max_value=64, step=16) if hp.Boolean('second_lstm',
                                                                    default = True) else None,
                 'dropout_2': hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1) if hp.Boolean('second_lstm',
                                                                    default = True) else None,
                 # Fully connected layer
                 'fc_units': hp.Int('fc_units', min_value=32, max_value=128, step=32),
                 'fc_dropout': hp.Float('fc_dropout', min_value=0.2, max_value=0.5, step=0.1)
                 }
            # Remove keys with None values (in case second_lstm is False)
            hp_dict = {k: v for k, v in hp_dict.items() if v is not None}

            # Learning rate using log-sampling (step argument omitted)
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

            # Create the model
            model = LSTMModel(input_shape, num_classes, hp_dict)

            return model, learning_rate
        except Exception as e:
            logger.error(f"Error during LSTM model building: {e}")
            return None, None

    @staticmethod
    def build_lstm_model_final(hp_dict, input_shape, num_classes):
        """
        Builds the final LSTM model using the best hyperparameters found.

        Args:
            hp_dict (dict): Dictionary with hyperparameter values.
            input_shape (tuple): (time_steps, features)
            num_classes (int): Number of output classes.

        Returns:
            LSTMModel: A PyTorch model instance.
        """
        try:
            model = LSTMModel(input_shape, num_classes, hp_dict)
            return model
        except Exception as e:
            logger.error(f"Error during final LSTM model building: {e}")
            return None

    @staticmethod
    def create_training_components(model, learning_rate, device='cuda'):
        """
        Create optimizer, loss function, and metric functions for training.

        Args:
            model (nn.Module): The model to train.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to use ('cuda' or 'cpu').

        Returns:
            tuple: (optimizer, loss_fn, metric_fns)
        """
        try:
            # Move model to device
            model = model.to(device)

            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Create loss function
            loss_fn = nn.CrossEntropyLoss()

            # Create metric functions
            def accuracy(y_pred, y_true):
                """Calculate accuracy."""
                # Apply softmax and get class predictions
                probs = F.softmax(y_pred, dim=1)
                predicted = torch.argmax(probs, dim=1)
                correct = (predicted == y_true).float()
                return correct.mean()

            def precision(y_pred, y_true):
                """Calculate precision (multi-class)."""
                # Apply softmax and get class predictions
                probs = F.softmax(y_pred, dim=1)
                predicted = torch.argmax(probs, dim=1)

                # One-hot encode predictions and true values
                num_classes = y_pred.size(1)
                y_pred_one_hot = F.one_hot(predicted, num_classes=num_classes).float()
                y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).float()

                # Calculate precision for each class
                true_positives = torch.sum(y_pred_one_hot * y_true_one_hot, dim=0)
                predicted_positives = torch.sum(y_pred_one_hot, dim=0)

                # Avoid division by zero
                precision_per_class = true_positives / (predicted_positives + 1e-7)

                # Return macro-averaged precision
                return torch.mean(precision_per_class)

            def recall(y_pred, y_true):
                """Calculate recall (multi-class)."""
                # Apply softmax and get class predictions
                probs = F.softmax(y_pred, dim=1)
                predicted = torch.argmax(probs, dim=1)

                # One-hot encode predictions and true values
                num_classes = y_pred.size(1)
                y_pred_one_hot = F.one_hot(predicted, num_classes=num_classes).float()
                y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).float()

                # Calculate recall for each class
                true_positives = torch.sum(y_pred_one_hot * y_true_one_hot, dim=0)
                actual_positives = torch.sum(y_true_one_hot, dim=0)

                # Avoid division by zero
                recall_per_class = true_positives / (actual_positives + 1e-7)

                # Return macro-averaged recall
                return torch.mean(recall_per_class)

            def f1_score(y_pred, y_true):
                """Calculate F1 score (multi-class)."""
                prec = precision(y_pred, y_true)
                rec = recall(y_pred, y_true)

                # Avoid division by zero
                return 2 * (prec * rec) / (prec + rec + 1e-7)

            # Bundle metrics into a dictionary
            metric_fns = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }

            return optimizer, loss_fn, metric_fns
        except Exception as e:
            logger.error(f"Error creating training components: {e}")
            return None, None, None

