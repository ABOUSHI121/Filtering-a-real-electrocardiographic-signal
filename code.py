import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from control import TransferFunction
from scipy.signal import butter, filtfilt, iirnotch, freqz, lfilter

######################################################################   Part 1.1    ######################################################################

# Read the Excel file
df = pd.read_excel("ECG1.xlsx")

# Define sample rate
sample_rate = 360  # Assuming a sample rate of 360 samples per second

# Compute real-time for each signal
for column in df.columns[3:]:  # Skip the first column (assuming the first column is the 'Sample Number')
    df[f'Real-Time ({column})'] = df['sample number'] * (1 / sample_rate)

# Write the updated DataFrame back to the Excel file
df.to_excel("real_time_1.xlsx", index=False)

# Read the Excel file
df = pd.read_excel("ECG2.xlsx")

# Define sample rate
sample_rate = 360  # Assuming a sample rate of 360 samples per second

# Compute real-time for each signal
for column in df.columns[3:]:  # Skip the first column (assuming the first column is the 'Sample Number')
    df[f'Real-Time ({column})'] = df['n'] * (1 / sample_rate)

# Write the updated DataFrame back to the Excel file
df.to_excel("real_time_2.xlsx", index=False)

# Define file paths
file_path_1 = 'real_time_1.xlsx'  # Update the path to your ECG1 file
file_path_2 = 'real_time_2.xlsx'  # Update the path to your ECG2 file

# Load ECG1 data
ecg1 = pd.read_excel(file_path_1)

######################################################################   Part 1.2    ######################################################################


# Plotting the ECG1 data
plt.figure(figsize=(10, 5))
plt.plot(ecg1['Real-Time (amplitude (mv).1)'], ecg1['amplitude (mv)'], linewidth=0.5)
plt.title('ECG Signal 1 1212478 & 1203331')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mv)')
plt.tight_layout()
plt.show()

# Load ECG2 data
ecg2 = pd.read_excel(file_path_2)

plt.figure(figsize=(10, 5))
plt.plot(ecg2['Real-Time (amplitude (mv).1)'], ecg2['amplitude (mv)'], linewidth=0.5)
plt.title('ECG Signal 2 1212478 & 1203331')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mv)')
plt.tight_layout()
plt.show()

######################################################################   Part 2.1    ######################################################################

# Frequency response of custom high-pass filter
b_custom_hp = [-1 / 32] + [0] * 14 + [1, -1] + [0] * 14 + [1 / 32]
a_custom_hp = [1, -1]

# Calculate the frequency response
w_custom_hp, h_custom_hp = freqz(b_custom_hp, a_custom_hp, worN=8000)

# Plot the frequency response of custom high-pass filter
plt.figure(figsize=(10, 5))
plt.plot(0.5 * sample_rate * w_custom_hp / np.pi, np.abs(h_custom_hp), 'b')
plt.title('Frequency Response of High-Pass Filter 1212478 & 1203331')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid()
plt.show()
print("The High-pass filter belongs to the FIR family.")

def hp_filter(signal):
    n = len(signal)
    hp = np.zeros(n)
    for i in range(32, n):
        hp[i] = hp[i - 1] - (1 / 32) * signal[i] + signal[i - 16] - signal[i - 17] + (1 / 32) * signal[i - 32]
    return hp


# Apply the high pass filter no the ECG1 and ECG2
ecg1_amplitude = ecg1['amplitude (mv)'].values  # Convert to numpy array
ecg1_hp = hp_filter(ecg1_amplitude)
ecg2_amplitude = ecg2['amplitude (mv)'].values  # Convert to numpy array
ecg2_hp = hp_filter(ecg2_amplitude)


# Plot ECG1 after high-pass filter
plt.figure(figsize=(10, 5))
plt.plot(ecg1['Real-Time (amplitude (mv).1)'], ecg1_hp, label='High-Pass Filtered Signal', linewidth=0.5, color='green')
plt.title('ECG1 Signal with High-Pass Filter 1212478 & 1203331')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot ECG2 after high-pass filter
plt.figure(figsize=(10, 5))
plt.plot(ecg2['Real-Time (amplitude (mv).1)'], ecg2_hp, label='High-Pass Filtered Signal', linewidth=0.5, color='green')
plt.title('ECG2 Signal with High-Pass Filter 1212478 & 1203331')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.tight_layout()
plt.show()





######################################################################   Part 2.2    ######################################################################


# Frequency response of custom low-pass filter
b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
a = [1, -2, 1]

# Calculate the frequency response
w, h = freqz(b, a, worN=8000)

# Plot the frequency response of the Low-Pass filter
plt.figure(figsize=(10, 5))
plt.plot(0.5 * sample_rate * w / np.pi, np.abs(h), 'b')
plt.title('Frequency Response of Low-Pass Filter 1212478 & 1203331')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid()
plt.show()

print("The low-pass filter belongs to the IIR family.")




# Custom low-pass filter based on given equation
def lp_filter(signal):
    n = len(signal)
    lp = np.zeros(n)
    for i in range(12, n):
        lp[i] = signal[i] - 2 * signal[i - 6] + signal[i - 12]
    return lp


#  Apply the high pass filter no the ECG1 and ECG2
ecg1_lp = lp_filter(ecg1_amplitude)
ecg2_lp = lp_filter(ecg2_amplitude)


# Plot ECG1 after high-pass filter
plt.figure(figsize=(10, 5))
plt.plot(ecg1['Real-Time (amplitude (mv).1)'], ecg1_lp, label='Low-Pass Filtered Signal', linewidth=0.5, color='green')
plt.title('ECG1 Signal with Low-Pass Filter 1212478 & 1203331')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot ECG2 after high-pass filter
plt.figure(figsize=(10, 5))
plt.plot(ecg2['Real-Time (amplitude (mv).1)'], ecg2_lp, label='Low-Pass Filtered Signal', linewidth=0.5, color='green')
plt.title('ECG2 Signal with Low-Pass Filter ')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.tight_layout()
plt.show()


######################################################################   Part 2.3    ######################################################################

#  High pass then low pass
ecg1_hp_lp = lp_filter(ecg1_hp)
ecg2_hp_lp = lp_filter(ecg2_hp)

#  High pass then low pass plot

# Plot ECG1 after high-pass filter then low-pass
plt.figure(figsize=(10, 5))
plt.plot(ecg1['Real-Time (amplitude (mv).1)'], ecg1_hp_lp, label='High-pass then Low-Pass Filtered Signal', linewidth=0.5, color='green')
plt.title('ECG1 Signal with High-pass then Low-Pass Filter 1212478 & 1203331')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot ECG2 after high-pass filter
plt.figure(figsize=(10, 5))
plt.plot(ecg2['Real-Time (amplitude (mv).1)'], ecg2_hp_lp, label='High-pass then Low-Pass Filtered Signal', linewidth=0.5, color='green')
plt.title('ECG2 Signal with High-pass then Low-Pass Filter ')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.tight_layout()
plt.show()

#  low-pass then high-pass
ecg1_lp_hp = hp_filter(ecg1_lp)
ecg2_lp_hp = hp_filter(ecg2_lp)

#  High pass then low pass plot

# Plot ECG1 after low-pass then high-pass
plt.figure(figsize=(10, 5))
plt.plot(ecg1['Real-Time (amplitude (mv).1)'], ecg1_lp_hp, label='Low-pass then High-Pass Filtered Signal', linewidth=0.5, color='green')
plt.title('ECG1 Signal with Low-pass then High-Pass Filter 1212478 & 1203331')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot ECG2 after low-pass then high-pass
plt.figure(figsize=(10, 5))
plt.plot(ecg2['Real-Time (amplitude (mv).1)'], ecg2_hp_lp, label='Low-pass then High-Pass Filtered Signal', linewidth=0.5, color='green')
plt.title('ECG2 Signal with Low-pass then High-Pass Filter ')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.tight_layout()
plt.show()

if np.array_equal(ecg1_lp_hp, ecg1_hp_lp):
    print("same")
else:
    print("not same")

if np.array_equal(ecg2_lp_hp, ecg2_hp_lp):
    print("same")
else:
    print("not same")