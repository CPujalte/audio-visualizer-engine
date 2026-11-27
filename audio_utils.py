# Audio Utils

import os
import subprocess
import wave
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_wav_from_mp4(input_path_mp4:    str,
                         output_path_wav:   str,
                         sample_rate:       int = 44100
                         ) -> None:
    """ Creates a .wav file from the audio of an .mp4. """

    cmd = ["ffmpeg",
           "-y", "-i",
           input_path_mp4,
           "-vn",
           "-acodec", 
           "pcm_s16le", "-ar",
           str(sample_rate),
           output_path_wav]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def render_video_frames_to_mp4(frames_temp_directory:  str,
                               input_path_wav:         str,
                               output_path_mp4:        str,
                               fps:                    int = 60
                               ) -> None:
    """ Gets all .png files from a folder and turns them into a .mp4. Max video lenght around 27 minutes. """
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_temp_directory, "frame_%05d.png"),
        "-i", input_path_wav,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path_mp4
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_energy_bands_matrix_from_wav(input_path_wav: str,
                                         n_bands: int,
                                         fps = 60,
                                         fft_size: int = 4096,
                                         min_freq: int = 60,
                                         max_freq: int = 16000
                                         ) -> np.ndarray:
    """ Returns a 2d array with the frequencies in each frame from a .wav file divided in n_bars bands. """

    # .wav variable extraction
    with wave.open(input_path_wav, "rb") as wav_file:
        n_channels: int = wav_file.getnchannels()
        sample_rate: int = wav_file.getframerate()
        n_frames: int = wav_file.getnframes()
        raw_bytes: np.ndarray = wav_file.readframes(n_frames)

    # From bytes to normalized samples between [-1, 1] in int16
    norm_samples = np.frombuffer(raw_bytes, np.int16) / 32768

    # Conversion into mono in case of stereo
    if n_channels == 2:
        norm_samples = norm_samples.reshape(-1, 2).mean(axis=1)

    # Number of frames in the video
    samples_per_frame: int = sample_rate // fps
    total_video_frames: int = len(norm_samples) // samples_per_frame

    # Magic frequency array via real fast fourier transform
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    # Hanning window to smoothen freqs
    hanning_window = np.hanning(fft_size)

    # Interval mask array definition in log scale
    group_edges = np.geomspace(min_freq, max_freq, n_bands + 1)
    group_masks = []
    group_centers = []  # Recoge los valores que representan la media entre el limite superior e inferior de cada mascara

    for index in range(n_bands):
        
        mask = (freqs >= group_edges[index]) & (freqs < group_edges[index+1])
        center = (group_edges[index] + group_edges[index+1]) / 2

        group_masks.append(mask)
        group_centers.append(center)

    # Matrix energy mean row generator
    energy_matrix = []
    for index in range(total_video_frames):

        # Choosing sample slice
        chunk_start:    int = index * samples_per_frame
        chunk_end:      int = chunk_start + fft_size
        chunk:          np.ndarray = norm_samples[chunk_start : chunk_end]

        # Index error prevention
        if len(chunk) < fft_size:
            break

        # Values in specific frame calculation
        magnitudes = np.abs(np.fft.rfft(chunk * hanning_window))
        frame_energy_row = []
        for mask, center in zip(group_masks, group_centers):
            if np.any(mask):    # Checks if array is empty to prevent NaN
                energy = np.mean(magnitudes[mask])
            else:               # Prediction via lineal interpolation if list is empty looking at closes rows
                energy = np.interp(center, freqs, magnitudes)
            frame_energy_row.append(energy)

        energy_matrix.append(np.array(frame_energy_row))

    return np.array(energy_matrix)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_rms_array_from_wav(input_path_wav:   str,
                               fps:              int = 60
                               ) -> np.ndarray:
    """ Returns an array with the RMS of each frame from a .wav. """

    # Extraccion de variables necesarias para el calculo
    with wave.open(input_path_wav, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw_bytes = wav_file.readframes(n_frames)

    # From bytes to normalized samples between [-1, 1] in int16
    norm_samples = np.frombuffer(raw_bytes, dtype=np.int16) / 32768

    # Conversion into mono in case of stereo
    if n_channels == 2:
        norm_samples = norm_samples.reshape(-1, 2).mean(axis=1)

    # RMS array obtention
    rms_list: list[float] = []
    samples_per_frame = sample_rate // fps
    total_video_frames = len(norm_samples) // samples_per_frame

    for i in range(total_video_frames):
        chunk = norm_samples[i * samples_per_frame : (i + 1) * samples_per_frame]
        rms = np.sqrt(np.mean(chunk**2))
        rms_list.append(rms)

    rms_array = np.array(rms_list)

    return rms_array

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def smoothen_ndarray_ema(raw_array:   np.ndarray,
                         alpha:       float = 0.2
                         ) -> np.ndarray:
    """ Smoothens all values in an ndarray via EMA. The lower the alpha, the higher the effect. """

    # Creates an ndarray of the same dimensions as raw_array and sets the first value
    smoothed_array = np.zeros_like(raw_array, dtype=float)
    smoothed_array[0] = raw_array[0]

    # Applies EMA to the whole array
    for i in range(1, len(raw_array)):
        smoothed_array[i] = raw_array[i] * alpha + smoothed_array[i-1] * (1-alpha)

    return smoothed_array

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def normalize_into_unit_interval(raw_array: np.ndarray
                                 ) -> np.ndarray:
    """ Normalizes all values from an ndarray into [0, 1] dividing by the maximum. """

    max_value = raw_array.max()
    if max_value == 0:              # Division by 0 protection
        return normalized_array
    
    normalized_array = raw_array / max_value

    return normalized_array

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def normalize_ndarray_log_scale(raw_array: np.ndarray,
                                ) -> np.ndarray:
    """ Applies ln to all values in an array and normalizes them between [0, 1]. """

    normalized_array = np.log1p(raw_array)
    normalized_array = normalize_into_unit_interval(normalized_array)

    return normalized_array