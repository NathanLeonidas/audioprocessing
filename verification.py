import librosa
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
import sounddevice as sd
import os

# Construire le chemin relatif vers le fichier audio
script_dir = os.path.dirname(os.path.abspath(__file__))  

# Charger le fichier audio avec soundfile
file_path = os.path.join(script_dir, 'audio_files', 'fluteircam.wav')

y, sr = librosa.load(file_path, sr=None)

# Extract the pitch using Parselmouth
sound = parselmouth.Sound(file_path)
pitch = sound.to_pitch(pitch_floor=50, pitch_ceiling=1000)  

# Get pitch values and filter out unvoiced parts (0 Hz)
pitch_values = pitch.selected_array['frequency']
pitch_values[pitch_values == 0] = np.nan  # Replace unvoiced parts with NaN for easier processing

# Extract time axis for the pitch
pitch_time = []
for i in range(pitch.get_number_of_frames()):
    pitch_time.append(pitch.get_time_from_frame_number(i + 1))

# Calculate mean pitch (global)
mean_pitch = np.nanmean(pitch_values)
print("Mean Pitch:", mean_pitch)

# Detect significant pitch variations
variation_threshold = 5  # Hz
significant_variations = []

for i in range(1, len(pitch_values)):
    if not np.isnan(pitch_values[i]) and not np.isnan(pitch_values[i - 1]):
        variation = abs(pitch_values[i] - pitch_values[i - 1])
        if variation > variation_threshold:
            significant_variations.append((pitch_time[i-1], variation))
            significant_variations.append((pitch_time[i], variation))

# Detect intervals where t and t+1 are separated by more than 0.03 seconds
interval_threshold = 0.03  # seconds
pitch_intervals = []
current_interval = []
for i in range(len(significant_variations) - 1):
    t1, _ = significant_variations[i]
    t2, _ = significant_variations[i + 1]
    if not current_interval:
        current_interval.append(t1)
    if t2 - t1 > interval_threshold:
        current_interval.append(t2)
        pitch_intervals.append(current_interval)
        current_interval = []
if current_interval:
    pitch_intervals.append(current_interval)

# Calculate the average pitch for each interval
interval_averages = []
for interval in pitch_intervals:
    start_time = interval[0]
    end_time = interval[1] if len(interval) > 1 else interval[0]
    start_index = next(i for i, t in enumerate(pitch_time) if t >= start_time)
    end_index = next(i for i, t in enumerate(pitch_time) if t >= end_time)
    
    # Extract pitch values for the interval
    interval_values = pitch_values[start_index:end_index + 1]
    
    # Ignore the first and last values of the interval
    if len(interval_values) > 2:  # Ensure there are at least 3 values to exclude 1st and last
        interval_values = interval_values[4:-4]
    
    # Calculate the average pitch for the remaining values
    interval_avg = np.nanmean(interval_values)
    interval_averages.append((start_time, end_time, interval_avg))

# Create the two plots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Full pitch contour
axs[0].plot(pitch_time, pitch_values, label="Pitch contour", color='blue')
axs[0].axhline(mean_pitch, color='red', linestyle='--', label=f"Mean Pitch: {mean_pitch:.2f} Hz")
significant_times = [item[0] for item in significant_variations]
significant_values = [pitch_values[pitch_time.index(time)] for time in significant_times]
axs[0].scatter(significant_times, significant_values, color='orange', label="Significant variations (>5 Hz)", zorder=5)
for start_time, end_time, interval_avg in interval_averages:
    axs[0].hlines(interval_avg, start_time, end_time, colors='green', linestyles='dotted', label=f"Avg: {interval_avg:.2f} Hz" if interval_avg else "")
axs[0].set_title("Pitch Contour with Significant Variations")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
#axs[0].legend()
axs[0].grid()

# Plot 2: Averages only
for start_time, end_time, interval_avg in interval_averages:
    axs[1].hlines(interval_avg, start_time, end_time, colors='green', linestyles='solid', linewidth=2, label=f"Avg: {interval_avg:.2f} Hz" if interval_avg else "")
axs[1].set_title("Interval Averages Only")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency (Hz)")
axs[1].grid()

# Remove duplicate legends in second plot
handles, labels = axs[1].get_legend_handles_labels()
unique_labels = set()
filtered_handles_labels = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        filtered_handles_labels.append((handle, label))
        unique_labels.add(label)
if filtered_handles_labels:
    filtered_handles, filtered_labels = zip(*filtered_handles_labels)
#    axs[1].legend(filtered_handles, filtered_labels)

# Display the plots
plt.tight_layout()
plt.show()

# Print interval averages
print("Interval Averages:")
for start_time, end_time, interval_avg in interval_averages:
    print(f"Interval: {start_time:.2f}s to {end_time:.2f}s, Average Pitch: {interval_avg:.2f} Hz")

def play_average_pitches(interval_averages, sampling_rate=44100):
    """
    Play the sounds corresponding to the average pitches of the intervals.
    Args:
        interval_averages (list): List of tuples (start_time, end_time, average_pitch).
        sampling_rate (int): Sampling rate for sound generation.
    """
    for start_time, end_time, avg_pitch in interval_averages:
        if not np.isnan(avg_pitch):  # Skip intervals without a valid pitch
            interval_duration = end_time - start_time  # Calculate the duration of the interval
            t = np.linspace(0, interval_duration, int(sampling_rate * interval_duration), endpoint=False)
            sound_wave = 0.5 * np.sin(2 * np.pi * avg_pitch * t)
            print(f"Playing pitch: {avg_pitch:.2f} Hz for {interval_duration:.2f} seconds")
            sd.play(sound_wave, samplerate=sampling_rate)
            sd.wait()  # Wait for the sound to finish


# Play the average pitches
play_average_pitches(interval_averages)
