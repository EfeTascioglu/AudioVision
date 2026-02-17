#!/usr/bin/env python3
"""
Play audio streamed over serial from ESP32.
Reads raw 32-bit, 3-channel PCM audio from COM5 and plays it in real-time.
"""

import serial
import struct
import threading
import numpy as np
from collections import deque
import time

try:
    import pyaudio
except ImportError:
    print("Error: pyaudio not installed. Install with: pip install pyaudio")
    exit(1)


# Audio settings (must match ESP32)
SAMPLE_RATE = 44100
CHANNELS = 3
BITS_PER_SAMPLE = 32
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8
FRAMES_PER_CHUNK = 256

# Serial settings
COM_PORT = "COM5"
BAUD_RATE = 115200
BUFFER_SIZE = 32 * 1024  # 32KB buffer


def read_serial(ser, buffer_queue):
    """Read raw audio data from serial port and queue it."""
    print(f"Reading from {COM_PORT} at {BAUD_RATE} baud...")
    try:
        while True:
            if ser.in_waiting > 0:
                # Read available data
                data = ser.read(min(ser.in_waiting, BUFFER_SIZE))
                if data:
                    buffer_queue.append(data)
    except Exception as e:
        print(f"Serial read error: {e}")


def play_audio_from_queue(buffer_queue):
    """Play audio data from the queue using pyaudio."""
    p = pyaudio.PyAudio()
    
    # Open audio stream (mono for testing, or stereo/multi-channel as needed)
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,  # Play only first channel for testing
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=FRAMES_PER_CHUNK,
    )
    # Data coming in is int_32. We need to convert to float32 in range [-1.0, 1.0] for pyaudio.
    # Assuming 24-bit signed integers, the maximum value is 2^23 - 1 = 8388607
    max_int = 2 ** (BITS_PER_SAMPLE - 1) - 1
    scale = 1.0 / max_int
    
    accumulated = b""
    samples_to_play = FRAMES_PER_CHUNK * BYTES_PER_SAMPLE 
    
    print(f"Playing audio ({SAMPLE_RATE} Hz, mono from first channel)...")
    
    try:
        while True:
            # Accumulate data from queue
            while buffer_queue and len(accumulated) < samples_to_play:
                accumulated += buffer_queue.popleft()
            
            # Play when we have enough data
            if len(accumulated) >= samples_to_play:
                chunk = accumulated[:samples_to_play]
                accumulated = accumulated[samples_to_play:]
                
                
                # Extract first channel (skip every 3rd sample after first)
                try:
                    samples = struct.unpack(f"<{FRAMES_PER_CHUNK * 1}i", chunk)
                    ch0_samples = [samples[i] for i in range(FRAMES_PER_CHUNK)]
                    print(f"First 5 samples in hex: {hex(ch0_samples[0]) if ch0_samples else 'None'}, {hex(ch0_samples[1]) if len(ch0_samples) > 1 else 'None'}, {hex(ch0_samples[2]) if len(ch0_samples) > 2 else 'None'}, {hex(ch0_samples[3]) if len(ch0_samples) > 3 else 'None'}, {hex(ch0_samples[4]) if len(ch0_samples) > 4 else 'None'}")
                    ch0_samples = [float(s) * scale for s in ch0_samples]  # Convert to float32 in range [-1.0, 1.0]
                    ch0_bytes = struct.pack(f"<{FRAMES_PER_CHUNK}f", *ch0_samples)
                    stream.write(ch0_bytes)
                    # print(f"Bytes: {len(chunk)}, Samples: {len(ch0_samples)}")
                except struct.error as e:
                    print(f"Parse error: {e}, skipping chunk")
            else:
                # Small sleep to avoid busy-waiting
                
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopping playback...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def test_sine_wave():
    """Generate and play a test 440Hz sine wave for 2 seconds."""
    import math
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=FRAMES_PER_CHUNK,
    )
    
    print("Playing test 440Hz sine wave for 2 seconds...")
    
    frequency = 440.0
    amplitude = 0.5
    duration = 0.5
    total_frames = int(SAMPLE_RATE * duration)
    
    for frame_idx in range(0, total_frames, FRAMES_PER_CHUNK):
        frames_to_gen = min(FRAMES_PER_CHUNK, total_frames - frame_idx)
        samples = []
        
        for i in range(frames_to_gen):
            t = (frame_idx + i) / SAMPLE_RATE
            sample = amplitude * math.sin(2.0 * math.pi * frequency * t)
            samples.append(sample)
        
        chunk = struct.pack(f"<{frames_to_gen}f", *samples)
        stream.write(chunk)
    
    print("Test tone finished")
    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
    import math
    
    # Test audio playback first
    print("Testing audio output with 440Hz sine wave...")
    test_sine_wave()
    print()
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to {COM_PORT}")
    except Exception as e:
        print(f"Failed to open {COM_PORT}: {e}")
        return
    
    buffer_queue = deque(maxlen=100)  # Max 100 chunks in buffer
    
    # Start serial read thread
    read_thread = threading.Thread(target=read_serial, args=(ser, buffer_queue), daemon=True)
    read_thread.start()
    
    # Play audio
    try:
        play_audio_from_queue(buffer_queue)
    finally:
        ser.close()
        print("Serial port closed")


if __name__ == "__main__":
    main()
