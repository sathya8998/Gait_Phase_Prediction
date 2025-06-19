<<<<<<< HEAD
# MindMove

poetry run pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitos.rrze.fau.de/n-squared-lab/software/demonstrations/mindmove.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitos.rrze.fau.de/n-squared-lab/software/demonstrations/mindmove/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
=======

========================================================
    EMG-DRIVEN GAIT PHASE CLASSIFICATION PIPELINE
========================================================

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
Supervisor: [Vlad Cnejevici]  
Advisor: [Prof. Dr. Alessandro Del Vecchio]

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

--------------------------------------------------------
  END OF FILE
--------------------------------------------------------

>>>>>>> 7ffb204 (Initial push of full Gait_Phase project structure with clean source files)
