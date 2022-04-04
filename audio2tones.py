#!/usr/bin/env python3
import sys
import logging
import datetime
from multiprocessing import Pool


import audiofile
import numpy as np
from scipy.fft import fft, fftfreq
from tqdm import tqdm
import matplotlib.pyplot as plt


mcus = {
    "ATmega328P": {
        # On the ATmega328P we have that the maximum PWM frequency in fastmode
        # is given by
        # f_pwm = f_clk_io/(n*256)
        # where n is one of the following prescale factors 1, 8, 32, 64, 128,
        # 256, 1024.
        "available_freqs": [16e6/(x*256) for x in [8, 32, 64, 128, 256, 1024]],
        # We never user 100% duty cycle, it wouldn't produce any sound
        "pwm_duty_cycle_max_value": 255 - 1,
        "playback_sampling_rate": int(10e3)
    }
}


def print_result(filename, data):
    with open(filename, "w") as f:
        f.write(f"int soundDataLen = {len(data)};\n")
        f.write("int soundData[][3] = {\n")
        for i, d in enumerate(data[:-1], start=1):
            duration, freq, mod = d
            f.write(f"{{0x{duration:04X},0x{freq:02X},0x{mod:02X}}},")
            if i % 3 == 0:
                f.write("\n")

        f.write(f"{{0x{data[-1][0]:04X},0x{data[-1][1]:02X},"
                f"0x{data[-1][2]:02X}}}\n}};")


def get_dominant_freq_and_module(window_chunk, sampling_rate,
                                 fourier_window_size):
    fourier_data = fft(window_chunk)
    # Original samples are taken at sampling_rate
    freqs = fftfreq(round(fourier_window_size), 1/sampling_rate)

    # Find the modulus
    modules = np.abs(fourier_data)
    max_module = np.max(modules)
    # Find the dominant frequency
    dominant_freq = freqs[np.argmax(modules)]

    return dominant_freq, max_module


def compress_pwm_audio(duration, frequency, modules):
    default_duration = round(duration[0])
    last_freq = frequency[0]
    last_module = modules[0]

    compressed_chunks = 0

    new_durations = []
    new_frequencies = [last_freq]
    new_modules = [last_module]

    for i, t in enumerate(duration[1:], 1):
        compressed_chunks += 1
        # Track volume or frequency changes
        if frequency[i] != last_freq or modules[i] != last_module:
            last_freq = frequency[i]
            last_module = modules[i]

            new_modules.append(last_module)
            new_frequencies.append(last_freq)
            new_durations.append(default_duration * compressed_chunks)

            compressed_chunks = 0

    new_durations.append(default_duration * compressed_chunks)
    return [*zip(new_durations, new_frequencies, new_modules)]


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s | %(message)s",
        level=logging.INFO
    )

    mcu = mcus["ATmega328P"]

    logging.info(f"Loading file: {sys.argv[1]}")
    samples, sampling_rate = audiofile.read(sys.argv[1])
    duration = len(samples[0])/sampling_rate

    logging.info(f"Loaded file: {sys.argv[1]}")
    logging.info(f"Channels: {len(samples)}")
    logging.info(f"Sampling rate: {sampling_rate} Hz")
    logging.info(f"Samples: {len(samples[0])}")
    logging.info(f"Duration: {datetime.timedelta(seconds=duration)}")

    # Merge all channels into one and drop the negative part of the PCM
    logging.info("Merging channels")
    merged_channels = np.maximum(
        np.amax(samples, axis=0),
        np.zeros(len(samples[0]))
    )


    # Define the number of samples to analyze for each window
    resampled_data_size = round(duration * mcu["playback_sampling_rate"])
    fourier_window_size = sampling_rate / mcu["playback_sampling_rate"]

    # Initialize default durations
    durations = np.full(
        resampled_data_size + 1,
        1/mcu["playback_sampling_rate"],
        dtype=np.int16
    )
    frequencies = np.empty(resampled_data_size + 1)
    modules = np.empty(resampled_data_size + 1)

    logging.info("Calculating FFT and resampling")
    with tqdm(desc="Processed windows", total=resampled_data_size) as pbar:

#        with Pool(8) as p:
#            print(p.map(f, [1, 2, 3]))
#            p.imap(
#                get_dominant_freq_and_module,
#
#            )
#

        for x, i in enumerate(np.array_split(merged_channels, resampled_data_size)):

            dominant_freq, max_module = get_dominant_freq_and_module(
                i, sampling_rate, fourier_window_size
            )

            # Append to the data
            frequencies[x] = dominant_freq
            modules[x] = max_module
            pbar.update(1)


    # Quantize the frequencies into the avaialble buckets
    bins = np.geomspace(
        frequencies.min() + 1,
        frequencies.max(),
        num=len(mcu["available_freqs"])
    )
    bins[0] = 0
    freq_indexes = np.digitize(frequencies, bins[::-1])
    plt.hist(frequencies, density=True, bins=6)
    plt.show()
    import pdb
    pdb.set_trace()

    quantized_freqs = np.fromiter(
        map(lambda x: mcu["available_freqs"][x]
            if x < len(mcu["available_freqs"]) else 0,
            freq_indexes
            ),
        dtype=np.int16
    )


    # Normalize wave pulses to PWM values
    compressed_range_modules = np.log(modules + 1)
    normalized_modules = np.interp(
        # Basic dynamic range compression
        compressed_range_modules,
        (0, compressed_range_modules.max()),
        (0, mcu["pwm_duty_cycle_max_value"])
    ).astype(np.int16)


    data = compress_pwm_audio(
        durations, quantized_freqs, normalized_modules
    )

    data = [*zip(durations, quantized_freqs, normalized_modules)]
    print_result("result.txt", data)


if __name__ == "__main__":
    main()
