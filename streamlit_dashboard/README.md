# SBMI – AI4Metabolomics

Understanding and monitoring metabolic processes is essential for studying diseases and therapeutic interventions. Nuclear Magnetic Resonance (NMR)–based methods, including Magnetic Resonance Spectroscopy (MRS) and related techniques, enable non-invasive investigation of metabolism *in vivo*, *in vitro*, and *ex vivo*.

This application provides an automated analysis pipeline for NMR spectra, allowing the detection and quantification of predefined metabolites. It combines classical model-based fitting approaches with a modular software design that prepares the system for future integration of AI-based methods.

---

## Features

- Automated Lorentzian peak fitting for time-resolved NMR spectra  
- Optional signal referencing for absolute quantification  
- Visualization of spectra, fitted curves, and kinetic traces  
- Stacked (waterfall) spectra visualization for temporal inspection  
- Batch processing of multiple spectrum files  
- Export of individual peak fits and reconstructed spectra  
- Streamlit-based graphical user interface  
- Automated Lorentzian peak fitting for time-resolved NMR spectra  
- Optional signal referencing for absolute quantification  
- Visualization of spectra, fitted curves, and kinetic traces  
- Stacked (waterfall) spectra visualization for temporal inspection  
- Batch processing of multiple spectrum files  
- Export of individual peak fits and reconstructed spectra  
- Streamlit-based graphical user interface  

## Comments from Andrey Pravdivtsev
#### Students WiSe25

- Added batch-mode processing.
- Removed the necessity for a reference file.
- Started working on using raw files instead of CSV (not finished).
- Made significant progress in adding deep learning (not finished and not implemented in the app).
- Explored line shapes beyond the Lorentzian shape, such as the Student-t shape and others (not implemented in the app).
- Docker installations sometimes do not work or require changes in the BIOS related to activation of the virtual environment.
- Implemented fitting of 13C thermal and hyperpolarized data (needs to be carefully checked).
- Implemented Docker installation.

#### Next generation should:
- Decrease data complexity by partial fitting using the idea of sparse data.
- Implement deep learning approaches.
- Implement more flexible line-shape fitting.
- Use raw data instead of preprocessed CSV files. 





---

## Run the Application

### Notes

- This application is a **research prototype** and not a fully production-ready system.  
- While the fitting pipeline itself is stable, some edge cases may still lead to runtime errors.  
- If an error occurs, please restart the application.  
- Always terminate the application via the terminal using **CTRL + C** to avoid orphaned processes and excessive memory usage.  

A Python version **≥ 3.10** is recommended.

---

## Prerequisites

### Install Git

To clone the repository, Git must be installed.  
Download Git from: https://git-scm.com/downloads

---

## Download the Repository

You can obtain the repository in one of the following ways.

### Option 1: Clone via Git (recommended)

```bash
git clone --depth 1 https://github.com/RATFIVE/SBMI-AI4metabolomics.git

```
### Option 2: Download as ZIP

1. Open the repository:  
   https://github.com/RATFIVE/SBMI-AI4metabolomics  
2. Click the green **Code** button  
3. Select **Download ZIP**

> **Note:** The download size may be larger due to legacy development files.

### Installation (Docker)

1. Check if you have ARM64 or AMD64 (x64) system:
Settings>System>About x64-based processor = AMD64
2. Install Docker: https://www.docker.com/ e.g. Windows for AMD64
3. Go to cmd and write wsl --update;
4. Run docker app (can skip login);
5. Open project in VS Code (file>open folder> navigate to SBMI_AI4Metaboloc...)
6. Open terminal in VS Code (Ctrl + J if it is not accessible on the bottom, or symbol close to the "close" button with highlighted bottom panel);
7. Write a command: docker build -t sbmi .  (copy the line with the "dot" starting with the docker)
8. After your docker image built use command: docker run -p 8501:8501 sbmi (docker run sbmi may not work for you)
9. Now you can close your VS Code and switch to docker. Open the container tab, click run, and then navigate to the port. A new browser window will open.
(for me http://localhost:8501/ works but http://0.0.0.0:8501 does not work).


If you have some more bugs, maybe check this one:
1. Check Intel VT-x (Virtualization Technology) and AMD SVM (Secure Virtual Machine, or AMD-V). It must be turned on (https://youtu.be/anP23TXuPU8?si=ODuRRsEet4zjOdG9 video about how to turn it on, It always different on all computers, but has the same names)


### Installation (direcdt)

It is strongly recommended to run the application inside a virtual environment

1. Navigate to the application directory

``` bash
cd path_to_your_download_dir/SBMI-AI4metabolomics-main/app

```

2. Create a virtual environment

```bash
python -m venv .SBMI

```

3. Activate the virtual environment

```bash
.SBMI\Scripts\activate

# Linus / MacOS
source .SBMI/bin/activate


```

4. Install required dependencies

```bash

pip install -r requirements.txt


```

### Start the applocation

```bash
streamlit run app.py

```


### Quit the application

CTRL + C


### Disclaimer

This software is intended for research and educational purposes only.
It is not certified for clinical use.