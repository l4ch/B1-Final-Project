import struct
import math
import cmath

def read_wave_file(filename):
    with open(filename, 'rb') as f:
        # Read the header
        f.read(22)  # Skip to number of channels
        num_channels = struct.unpack('<H', f.read(2))[0]
        sample_rate = struct.unpack('<I', f.read(4))[0]
        f.read(6)  # Skip to bits per sample
        bits_per_sample = struct.unpack('<H', f.read(2))[0]
        f.read(4)  # Skip to data chunk size
        data_size = struct.unpack('<I', f.read(4))[0]
        
        # Read the data
        raw_data = f.read(data_size)
        
        # Convert raw data to integers
        bytes_per_sample = bits_per_sample // 8
        total_samples = data_size // bytes_per_sample
        fmt = '<' + 'i' * total_samples
        integer_data = struct.unpack(fmt, raw_data)
        
        # Extract the left channel data if stereo
        left_channel_data = integer_data[0::num_channels]
        
        return left_channel_data, sample_rate

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [cmath.exp(-2j * math.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def fft_nolib(filename):
    audio_data, sample_rate = read_wave_file(filename)
    transformed = fft(audio_data)
    magnitudes = [abs(x) for x in transformed]
    
    # Display results on a logarithmic scale between 0 and 20,000Hz
    max_freq = 20000
    max_index = int(max_freq * len(magnitudes) / sample_rate)
    for i in range(max_index):
        freq = i * sample_rate / len(magnitudes)
        magnitude = magnitudes[i]
        log_scale = math.log10(freq + 1)
        print(f"Frequency: {freq:.2f} Hz, Magnitude: {magnitude:.2f}, Log Scale: {log_scale:.2f}")

# Example usage
#fft_nolib('Audio-Samples/sine.wav')