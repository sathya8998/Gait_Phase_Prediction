import pandas as pd

def run_merging_pipeline():
    try:
        print("Merging muscle synergy and predicted gait stages...")

        df_synergy = pd.read_csv("C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\muscle_synergy_activations.csv")
        df_predicted = pd.read_csv("C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\predicted_stride_stages_final.csv")

        df_merged = pd.merge(df_synergy, df_predicted[['Frame', 'PredictedStage']], on='Frame', how='left')

        STRIDE_INTERVALS = [
            (89108, 92245), (92246, 95175), (95176, 97447),
            (97448, 100575), (100576, 103623), (103624, 106831),
            (106832, 109945), (109946, 112821), (112822, 115905),
            (115906, 119045), (119046, 122367), (122368, 126179),
            (126180, 128417), (128418, 131367), (131368, 135647),
            (135648, 137665), (137666, 141871), (141872, 145001),
            (145002, 147199), (147200, 150373), (150374, 153445),
            (153446, 156627), (156628, 159901)
        ]
        mask = pd.Series(False, index=df_merged.index)
        for start, end in STRIDE_INTERVALS:
            mask |= (df_merged['Frame'] >= start) & (df_merged['Frame'] <= end)

        df_stride = df_merged[mask].copy()
        df_clean = df_stride.dropna()

        output_path = "C:\\Users\\ek23yboj\\PycharmProjects\\PythonProject\\Gait_Phase\\Data\\final_labels_cleaned_merged.csv"
        df_clean.to_csv(output_path, index=False)
        print(f"Merged file saved to: {output_path}")

    except Exception as e:
        print(f"Error during merging pipeline: {e}")
