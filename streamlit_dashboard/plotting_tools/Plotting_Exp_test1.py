import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 # pip install opencv-python
from glob import glob
import shutil
import numpy as np

# === File paths ===
file1 = r".\Data\FA_20240205_2H_yeast_Acetone-d6_3.csv"
kinetics_file = r".\app\output\FA_20240205_2H_yeast_Acetone-d6_3.csv_output\kinetics_mmol.csv"

select_plot={"number":1,"orientation":"h"} 

# === Load data ===
df1 = pd.read_csv(file1)
print(df1.head())

# === Total Repetition Time ===
TRtot = 11.5 * 8  # seconds

# === List of experiment IDs to plot ===
exp_ids = range(1, 65)  # Adjust as needed

# === Create output directory for composite frames ===
composite_dir = "composite_frames"
os.makedirs(composite_dir, exist_ok=True)

# === Prepare kinetics data ===
if os.path.exists(kinetics_file):
    kinetics_df = pd.read_csv(kinetics_file)
    print("\nLoaded kinetics_mmol.csv:")
    print(kinetics_df.head())
    
    time_points_min = kinetics_df["Time_Step"] * TRtot / 60 # convert seconds to minutes
    
    fumarate = kinetics_df['ReacSubs'] if 'ReacSubs' in kinetics_df.columns else kinetics_df.iloc[:, 1]
    hdo = kinetics_df['Water'] if 'Water' in kinetics_df.columns else kinetics_df.iloc[:, 2]
    malate = kinetics_df['Metab1'] if 'Metab1' in kinetics_df.columns else kinetics_df.iloc[:, 3]
else:
    print(f"❌ Kinetics file not found: {kinetics_file}")
    exit()

frame_count = 0

# === Loop through experiments ===
for exp_id in exp_ids:
    if exp_id > len(time_points_min):
        break  # Stop if we've reached the end of kinetics data
        
    exp_id_name = f"FA_20231123_2H_yeast_1.12.ser#{exp_id}"
    file2 = fr".\app\output\FA_20231123_2H Yeast_Fumarate-d2_12 .csv_output\substance_fits\substance_fit_{exp_id}.csv"
    
    if not os.path.exists(file2):
        print(f"File not found: {file2}")
        continue

    df2 = pd.read_csv(file2)
    
    x = df1['2H chemical shift (ppm)']
    if exp_id_name not in df1.columns:
        print(f"Column not found in df1: {exp_id_name}")
        continue

    raw = df1[exp_id_name]
    fitted = df2["ReacSubs"] + df2["Water"] + df2["Metab1"]
    residual = raw - fitted
    
    # Normalize by max raw intensity
    max_raw = max(raw)
    raw_normalized = raw / max_raw
    fitted_normalized = fitted / max_raw
    residual_normalized = residual / max_raw
    metab1_normalized = df2["Metab1"] / max_raw
    water_normalized = df2["Water"] / max_raw
    reacsubs_normalized = df2["ReacSubs"] / max_raw

    # fig = plt.figure(figsize=(15, 8))
    #fig = plt.figure(figsize=(8, 12))
    fig = plt.figure(figsize=(8, 6))
    
    # First subplot - Spectrum
    #ax1 = plt.subplot(1, 2, 1)
    #ax1 = plt.subplot(2, 1, 1)
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(x, raw_normalized, label='Raw data')
    ax1.plot(df2['x'], np.column_stack([reacsubs_normalized, water_normalized, metab1_normalized]), 
             label=['Fumarate', 'HDO', 'Malate'])
    ax1.plot(x, residual_normalized, label='Residual', linestyle='--', color='black')
    
    current_time_min = exp_id * TRtot / 60  # convert to minutes
    ax1.set_title(f'Experiment # {exp_id}, time = {current_time_min:.2f} min', fontsize=20)
    
    ax1.set_xlabel(r'$^{2}$H chemical shift (ppm)', fontsize=20)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=20)
    ax1.grid(False)
    ax1.set_xlim(9.5, 1)
    ax1.set_ylim(bottom=-0.1, top=2)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.legend(fontsize=20)
    
    # Second subplot - Kinetics
    #ax2 = plt.subplot(1, 2, 2)
    if False:
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(time_points_min[:exp_id], fumarate[:exp_id], label='Fumarate', color='orange', marker='o')
        ax2.plot(time_points_min[:exp_id], hdo[:exp_id], label='HDO', color='green', marker='s')
        ax2.plot(time_points_min[:exp_id], malate[:exp_id], label='Malate', color='red', marker='^')
        
        ax2.set_xlabel('Time (min)', fontsize=20)  # Changed to minutes
        ax2.set_ylabel('Concentration (mmol/L)', fontsize=20)
        ax2.grid(False)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(bottom=-1, top=40)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.legend(fontsize=20)
        
        plt.tight_layout()

    # Save frame
    composite_path = os.path.join(composite_dir, f"composite_{frame_count:03d}.png")
    plt.savefig(composite_path)
    plt.close()
    frame_count += 1

print(f"\nSaved {frame_count} composite frames to '{composite_dir}'")

# === Make a video from the composite frames ===
output_video = "combined_plot_video.mp4"
frame_rate = 10  # fps

images = sorted(glob(os.path.join(composite_dir, "composite_*.png")))
if not images:
    print("No frames found to make a video.")
    exit()

# Get image size from first frame
frame = cv2.imread(images[0])
height, width, _ = frame.shape

# Define video writer
video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

for image in images:
    video.write(cv2.imread(image))

video.release()
print(f"✅ Combined video saved as '{output_video}'")

# === Optional: Clean up frames ===
delete_frames = True  # Change to False to keep frames
if delete_frames:
    shutil.rmtree(composite_dir)
    print(f"🧹 Deleted temporary folder: {composite_dir}")
