import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from Gait_Phase.main.main import main_pipeline
import torch
from Gait_Phase.preprocessing.Preprocessing_wrapped import align_emg_data_pipeline as preprocessing
from Gait_Phase.nmf_channels_reduction.NMF_Channels_Reduction_wrapped import run_nmf_channel_reduction_pipeline
from Gait_Phase.fusion_hmm.Fusion_HMM_wrapped import run_fusion_hmm_pipeline
from Gait_Phase.merging.merging_pipeline import run_merging_pipeline
from Gait_Phase.hyperparameter_tuning.HyperParameter_Tuning import hyperparameter_tuning
from Gait_Phase.attention_lstm.Attention_Layer_Bi_Lstm_wrapped import LSTMModelBuilder
from Gait_Phase.cnn_model.CNN_wrapped import CNNModelBuilder
from Gait_Phase.transformers.Transformers_wrapped import TransformerModelBuilder
from Gait_Phase.gpu_utils import configure_gpu

def run_all():
    # Define base path for saving models
    save_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\"
    global BEST_STRATEGIES_PATH
    print("== Configuring GPU ==")
    configure_gpu()

    print("== Step 1: Preprocessing ==")
    preprocessing()

    print("== Step 2: NMF Channel Reduction ==")
    run_nmf_channel_reduction_pipeline()

    print("== Step 3: Fusion HMM ==")
    run_fusion_hmm_pipeline()

    print("== Step 4: Merging ==")
    run_merging_pipeline()

    # Load best hyperparameters
    best_strategies = {}
    try:
        with open(BEST_STRATEGIES_PATH, "r") as file:
            best_strategies = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {BEST_STRATEGIES_PATH}")
    except Exception as e:
        print(f"Error loading best strategies: {e}")

    print("== Step 5: LSTM ==")
    lstm_model_builder = LSTMModelBuilder()
    lstm_hp_dict = best_strategies.get("LSTM", {}).get("Best_Hyperparameters")
    if lstm_hp_dict:
        lstm_model = lstm_model_builder.build_lstm_model_final(lstm_hp_dict, input_shape=(128, 8), num_classes=5)
        if lstm_model:
            lstm_save_path = os.path.join(save_path, "lstm_best_model.pth")
            torch.save(lstm_model.state_dict(), lstm_save_path)
            print(f"LSTM model saved to {lstm_save_path}")

    print("== Step 6: CNN ==")
    cnn_model_builder = CNNModelBuilder()
    cnn_hp_dict = best_strategies.get("CNN", {}).get("Best_Hyperparameters")
    if cnn_hp_dict:
        cnn_model = cnn_model_builder.build_cnn_model_final(cnn_hp_dict, input_shape=(128, 8), num_classes=5)
        if cnn_model:
            cnn_save_path = os.path.join(save_path, "cnn_best_model.pth")
            torch.save(cnn_model.state_dict(), cnn_save_path)
            print(f"CNN model saved to {cnn_save_path}")

    print("== Step 7: Transformer ==")
    transformer_model_builder = TransformerModelBuilder()
    transformer_hp_dict = best_strategies.get("Transformer", {}).get("Best_Hyperparameters")
    if transformer_hp_dict:
        transformer_model = transformer_model_builder.build_transformer_model_final(
            transformer_hp_dict, input_shape=(128, 8), num_classes=5
        )
        if transformer_model:
            transformer_save_path = os.path.join(save_path, "transformer_best_model.pth")
            torch.save(transformer_model.state_dict(), transformer_save_path)
            print(f"Transformer model saved to {transformer_save_path}")

    print("== Step 8: Main Pipeline ==")
    main_pipeline()

if __name__ == "__main__":
    run_all()
