--------------------------------------------------------
    EMG-DRIVEN GAIT PHASE CLASSIFICATION PIPELINE
--------------------------------------------------------

Welcome to the readme file for the project.

--------------------------------------------------------
  SYSTEM REQUIREMENTS
--------------------------------------------------------
Please ensure your system meets the following requirements:

- Python 3.10+
- PyTorch with CUDA support (recommended for training with GPU)
- At least 16 GB of RAM (recommended for deep learning workloads)
- Minimum 60-70 GB of free disk space for processed EMG and model files

--------------------------------------------------------
  INSTALLATION GUIDE
--------------------------------------------------------

It is recommended to use a virtual environment to avoid dependency conflicts.

1. **Create and activate a virtual environment**

```bash
python3 -m venv gait_phase_env
source gait_phase_env/bin/activate   # On Windows: gait_phase_env\Scripts\activate
```

2. **Install the required dependencies**

```bash
pip install -r requirements.txt
```

If using Poetry instead:

```bash
poetry install
```

--------------------------------------------------------
  EXECUTING THE PIPELINE
--------------------------------------------------------

To run the **entire pipeline** from preprocessing to model training and evaluation:

```bash
python run_all.py
```

- This script acts as a **one-click wrapper** for the complete workflow.
- It calls:
  - Preprocessing
  - Synergy extraction  - Channel reduction
  - HMM-based fusion using marker and synergy features
  - Deep learning model training (CNN, BiLSTM, Transformer)
  - Final output generation and interpretation

---

Alternatively, individual stages can be run manually using:

1. Preprocess Raw EMG and Marker Data

For Vicon-based data:
   ```bash
python preprocessing/Preprocessing_wrapped.py
   ```

For OptiTrack-based data:
   ```bash
python preprocessing/Preprocessing_wrapped_overground.py
   ```

2. Extract Muscle Synergies and Reduce Channels
   ```bash
   python nmf_channels_reduction/NMF_Channels_Reduction_wrapped.py
   ```
   
Predict TA Activation from VL Features
   ```bash
   python nmf_channels_reduction/ta_from_vl_predictor.py
   ```

3. Apply HMM-Based Fusion
   ```bash
   python fusion_hmm/Fusion_HMM_wrapped.py
   ```

4. Train Deep Learning Models
   ```bash
   python main/main.py
   ```

--------------------------------------------------------
  FILE AND DATA FOLDER CONTEXT
--------------------------------------------------------

Folders:

- `Data/` : Processed participant-specific EMG + marker files
- `models/` : Trained `.pt` files
- `preprocessing/` : Scripts to clean and structure raw data
- `nmf_channels_reduction/` : NMF + channel reduction logic
- `main/` : Pipeline controller script
- `fusion_hmm/` : Post-processing refinement using HMM
- `cnn_model/`, `transformers/`, `attention_lstm/` : Deep learning model definitions

Files:

- `run_all.py` : **One-click wrapper to execute full pipeline**
- `requirements.txt` : Python dependency list
- `.gitignore` : Excludes model and data folders

--------------------------------------------------------
  FILE ORGANIZATION
--------------------------------------------------------

Organize your directory as follows:

```text
Gait_Phase/
├── Data/                         
├── models/                        
├── preprocessing/
├── nmf_channels_reduction/
├── cnn_model/
├── transformers/
├── attention_lstm/
├── fusion_hmm/
├── main/
├── evaluation_utils/
├── analysis_utils/
├── run_all.py
├── requirements.txt
├── .gitignore
```

--------------------------------------------------------
  DATA STRUCTURE
--------------------------------------------------------

Example participant folder:
```text
Data/
├── 001/
│   ├── emg_cleaned.csv
│   └── markers_aligned.csv
├── 002/
│   ├── ...
```

Data folders are excluded from Git and only used locally.

---

- Subject folders (001–007)
- `emg_cleaned.csv` and `markers_aligned.csv`
- Sampling rate (EMG: 2000 Hz, Markers: 125 Hz/Markers: 100 Hz)

--------------------------------------------------------
  CREDITS
--------------------------------------------------------
Developed as part of the Master’s Thesis at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).

Author: Sathyanarayanan Dhorali  
Advisor: [Vlad Cnejevici]  
Supervisor: [Prof. Dr. Alessandro Del Vecchio]

For more details, refer to the Master’s Thesis Document.


--------------------------------------------------------
  FEATURE-WISE MARKER USAGE: VICON vs. OPTITRACK
--------------------------------------------------------

| Feature Type         | Vicon Markers Used                                                     | OptiTrack Markers Used             |
|----------------------|------------------------------------------------------------------------|------------------------------------|
| Position             | RTHI, LTHI, RKNE, LKNE, RTIB, LTIB, RANK, LANK, RHEE, LHEE, RTOE, LTOE | RGT, LGT, RLE, LLE, RLM, LLM       |
| Velocity             | RTHI, LTHI, RKNE, LKNE, RTIB, LTIB, RANK, LANK, RHEE, LHEE, RTOE, LTOE | RGT, LGT, RLE, LLE, RLM, LLM       |
| Acceleration         | RTHI, LTHI, RKNE, LKNE, RTIB, LTIB, RANK, LANK, RHEE, LHEE, RTOE, LTOE | RGT, LGT, RLE, LLE, RLM, LLM       |
| Acceleration Magnitude | RTHI, LTHI, RKNE, LKNE, RTIB, LTIB, RANK, LANK, RHEE, LHEE, RTOE, LTOE | RGT, LGT, RLE, LLE, RLM, LLM       |
| Angular Momentum     | RTHI, LTHI, RKNE, LKNE, RTIB, LTIB, RANK, LANK, RHEE, LHEE, RTOE, LTOE | RGT, LGT, RLE, LLE                 |
| Joint Angle Triplets | RTHI-RKNE-RTIB, LTHI-LKNE-LTIB                                         | RGT-RLE-RLM, LGT-LLE-LLM           |
| Distance Pairs       | LTHI-RTHI, LKNE-RKNE, LTIB-RTIB, LANK-RANK, LHEE-RHEE, LTOE-RTOE       | RLE-LLE, RGT-LGT, RLE-RLM, LLE-LLM, RGT-RLE, LGT-LLE |

