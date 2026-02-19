from __future__ import annotations

import argparse
import base64
import io
import json
import math
import socket
import struct
import threading
import time
import urllib.request
import wave
from collections import deque
from queue import Queue
from typing import Dict, Optional, Tuple

import numpy as np
from flask import Flask, Response, jsonify, request, send_file

from typing import List

from audio_transcription.whisper_live_kit.audio_main import run as run_audio_main


app = Flask(__name__)

# UDP settings
UDP_PORT_MIC0 = 30001
UDP_PORT_MIC1 = 30002
UDP_PORT_MIC2 = 30003
udp_sockets = []
udp_threads = []
packet_counters = {0: 0, 1: 0, 2: 0}  # Track packets received per mic
packet_errors = {0: 0, 1: 0, 2: 0}    # Track errors per mic

latest_lock = threading.Lock()
latest_mic0: Dict[str, object] = {
    "device_id": "udp-mic0",
    "sample_rate": None,
    "channels": 1,
    "bits": 32,
    "format": None,
    "timestamp": None,
    "data": None,
    "rms": 0.0,
}
latest_mic1: Dict[str, object] = {
    "device_id": "udp-mic1",
    "sample_rate": None,
    "channels": 1,
    "bits": 32,
    "format": None,
    "timestamp": None,
    "data": None,
    "rms": 0.0,
}
latest_mic2: Dict[str, object] = {
    "device_id": "udp-mic2",
    "sample_rate": None,
    "channels": 1,
    "bits": 32,
    "format": None,
    "timestamp": None,
    "data": None,
    "rms": 0.0,
}

subscribers: "list[Queue]" = []

# Circular buffer for last 5 seconds of audio - separate per-mic for 3-channel reassembly
audio_buffer_lock = threading.Lock()
audio_buffer_mic0: deque = deque(maxlen=500)  # Channel 0 (left/I2S0)
audio_buffer_mic1: deque = deque(maxlen=500)  # Channel 1 (right/I2S0)
audio_buffer_mic2: deque = deque(maxlen=500)  # Channel 2 (mono/I2S1)
buffer_sample_rate = 48000  # ESP32 uses 48kHz
buffer_channels = 3         # 3-channel interleaved output
buffer_bits = 32            # ESP32 sends 32-bit samples
# Legacy audio_buffer for backward compatibility (mixed packets)
audio_buffer: deque = deque(maxlen=500)


def _is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.close()
        return True
    except OSError:
        return False


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _compute_rms(data: bytes, channels: int, bits_per_sample: int = 16) -> Tuple[float, ...]:
    if not data or channels <= 0:
        return tuple()
    bytes_per_sample = bits_per_sample // 8
    samples = len(data) // bytes_per_sample
    frames = samples // channels
    if frames == 0:
        return tuple()

    acc = [0.0] * channels
    idx = 0
    for _ in range(frames):
        for ch in range(channels):
            sample = int.from_bytes(data[idx : idx + bytes_per_sample], "little", signed=True)
            acc[ch] += float(sample * sample)
            idx += bytes_per_sample

    return tuple((acc[ch] / frames) ** 0.5 for ch in range(channels))


def udp_receiver(port: int, mic_id: int):
    """UDP receiver thread for a single microphone with comprehensive diagnostics."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # For Windows compatibility, also try SO_REUSEPORT if available
    if hasattr(socket, 'SO_REUSEPORT'):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (OSError, AttributeError):
            pass
    
    # Try to bind with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            sock.bind(("0.0.0.0", port))
            sock.settimeout(1.0)
            print(f"UDP receiver for MIC{mic_id} listening on port {port}")
            break
        except OSError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"[MIC{mic_id}] Failed to bind port {port} (attempt {attempt+1}/{max_retries}): {e}")
                print(f"[MIC{mic_id}] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[MIC{mic_id}] FATAL: Could not bind port {port} after {max_retries} attempts: {e}")
                print(f"[MIC{mic_id}] Port may be in use or requires elevated privileges")
                packet_errors[mic_id] += 1000  # Mark as critical error
                return
    
    # Select the appropriate latest_mic dictionary
    if mic_id == 0:
        latest_mic = latest_mic0
    elif mic_id == 1:
        latest_mic = latest_mic1
    else:
        latest_mic = latest_mic2
    
    last_log_time = time.time()
    
    while True:
        try:
            data, addr = sock.recvfrom(65536)  # Max UDP packet size
            packet_size = len(data)
            
            if packet_size > 0:
                # Validate packet size (should be multiple of 4 for 32-bit samples)
                if packet_size % 4 != 0:
                    packet_errors[mic_id] += 1
                    print(f"[MIC{mic_id}] Invalid packet size: {packet_size} bytes (not multiple of 4)")
                    continue
                
                num_samples = packet_size // 4
                
                # Decode samples for validity check
                try:
                    samples = struct.unpack(f'<{num_samples}i', data)  # Little-endian 32-bit signed
                    # Check for reasonable sample values (not all zeros, not all 0xaaaabbbbccccdddd)
                    min_val, max_val = min(samples), max(samples)
                    
                    # Log packet statistics every 3 seconds
                    now = time.time()
                    if now - last_log_time >= 3.0:
                        print(f"[MIC{mic_id}] Port {port}: packets={packet_counters[mic_id]}, "
                              f"errors={packet_errors[mic_id]}, size={packet_size}B, "
                              f"samples={num_samples}, min={min_val}, max={max_val}, "
                              f"from {addr}")
                        last_log_time = now
                    
                except struct.error as e:
                    packet_errors[mic_id] += 1
                    print(f"[MIC{mic_id}] Struct unpack error: {e}")
                    continue
                
                packet_counters[mic_id] += 1
                
                # Store in buffer for this mic channel (both legacy and per-mic)
                with audio_buffer_lock:
                    audio_buffer.append(data)  # Legacy mixed buffer
                    # Store in per-mic buffer for 3-channel reassembly
                    if mic_id == 0:
                        audio_buffer_mic0.append(data)
                    elif mic_id == 1:
                        audio_buffer_mic1.append(data)
                    else:
                        audio_buffer_mic2.append(data)
                
                # Compute RMS for this packet (32-bit samples, 1 channel)
                rms_values = _compute_rms(data, 1, 32)
                rms = rms_values[0] if rms_values else 0.0
                
                # Encode data as base64 for web UI
                data_b64 = base64.b64encode(data).decode('utf-8')
                
                # Update latest packet info for this specific mic
                with latest_lock:
                    latest_mic["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    latest_mic["data"] = data_b64  # Store as base64
                    latest_mic["sample_rate"] = buffer_sample_rate
                    latest_mic["rms"] = rms
                
                # Notify subscribers with combined data
                packet_info = {
                    "device_id": latest_mic["device_id"],
                    "sample_rate": latest_mic["sample_rate"],
                    "channels": 1,
                    "bits": 32,
                    "timestamp": latest_mic["timestamp"],
                    "data": data_b64,  # Send base64
                    "rms": rms,
                }
                for queue in list(subscribers):
                    queue.put(packet_info)
        except socket.timeout:
            continue
        except Exception as e:
            packet_errors[mic_id] += 1
            print(f"UDP receiver MIC{mic_id} error: {e}")
            time.sleep(0.1)


@app.after_request
def add_cors_headers(resp: Response) -> Response:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,X-Device-Id,X-Sample-Rate,X-Channels,X-Bits,X-Format"
    return resp


@app.route("/")
def index() -> Response:
    html = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>AudioVision Debug</title>
        <style>
          body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 1rem; margin: 0; }
          h1 { margin: 0 0 1rem 0; color: #4ec9b0; }
          .status { padding: 0.5rem; background: #2d2d30; margin-bottom: 1rem; border-left: 3px solid #007acc; }
          .section { background: #252526; padding: 1rem; margin-bottom: 1rem; border: 1px solid #3e3e42; }
          .section h2 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #569cd6; }
          .data { display: grid; grid-template-columns: 150px auto; gap: 0.5rem; }
          .label { color: #9cdcfe; }
          .value { color: #ce9178; }
          .log { max-height: 300px; overflow-y: auto; font-size: 0.875rem; }
          .log-entry { padding: 0.25rem; border-bottom: 1px solid #3e3e42; }
          .log-entry:hover { background: #2d2d30; }
          .rms { display: flex; gap: 1rem; }
          .rms-item { flex: 1; }
          .rms-label { font-size: 0.75rem; color: #9cdcfe; margin-bottom: 0.25rem; }
          .rms-value { font-size: 1.25rem; color: #4ec9b0; font-weight: bold; margin-bottom: 0.5rem; }
          .rms-bar-bg { height: 120px; background: #3e3e42; border-radius: 4px; position: relative; overflow: hidden; }
          .rms-bar-fill { width: 100%; background: linear-gradient(to top, #4ec9b0, #569cd6, #c586c0); 
                          position: absolute; bottom: 0; transition: height 0.1s ease-out; }
          .controls { margin-top: 1rem; text-align: center; display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; }
          .btn { padding: 0.75rem 1.5rem; background: #007acc; color: #fff; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; font-family: monospace; }
          .btn.secondary { background: #3a3d41; }
          .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        </style>
      </head>
      <body>
        <h1>AudioVision Debug</h1>
        <div class="status" id="status">Waiting for data...</div>
        
        <div class="section">
          <h2>Latest Packet</h2>
          <div class="data">
            <div class="label">Device ID:</div><div class="value" id="device">-</div>
            <div class="label">Sample Rate:</div><div class="value" id="rate">-</div>
            <div class="label">Channels:</div><div class="value" id="channels">-</div>
            <div class="label">Bits:</div><div class="value" id="bits">-</div>
            <div class="label">Bytes:</div><div class="value" id="bytes">-</div>
            <div class="label">Timestamp:</div><div class="value" id="ts">-</div>
            <div class="label">Packets Received:</div><div class="value" id="count">0</div>
          </div>
        </div>

        <div class="section">
          <h2>RMS Levels</h2>
          <div class="rms" id="rms"></div>
          <div class="controls">
            <select id="listenMode" class="btn secondary" style="width: auto; padding: 0.75rem;">
              <option value="3ch">Listen: 3-Channel</option>
              <option value="mic0">Listen: Mic 0 Only</option>
              <option value="mic1">Listen: Mic 1 Only</option>
              <option value="mic2">Listen: Mic 2 Only</option>
            </select>
            <button id="liveBtn" class="btn secondary">Listen Live</button>
            <button id="saveBtn" class="btn">Save Last 5 Seconds</button>
            <button id="textBtn" class="btn secondary">Export as Text</button>
            <button id="diagBtn" class="btn secondary">Check Diagnostics</button>
          </div>
        </div>

        <div class="section">
          <h2>Packet Diagnostics</h2>
          <div class="data" id="diagnostics">
            <div class="label">Total Packets:</div><div class="value" id="total-packets">-</div>
            <div class="label">MIC0 Packets:</div><div class="value" id="mic0-packets">-</div>
            <div class="label">MIC1 Packets:</div><div class="value" id="mic1-packets">-</div>
            <div class="label">MIC2 Packets:</div><div class="value" id="mic2-packets">-</div>
            <div class="label">Total Errors:</div><div class="value" id="total-errors">-</div>
            <div class="label">Last MIC0:</div><div class="value" id="last-mic0">-</div>
            <div class="label">Last MIC1:</div><div class="value" id="last-mic1">-</div>
            <div class="label">Last MIC2:</div><div class="value" id="last-mic2">-</div>
          </div>
        </div>

        <div class="section">
          <h2>Packet History (Last 20)</h2>
          <div class="log" id="log"></div>
        </div>

        <script>
          let packetCount = 0;
          let logEntries = [];
          const maxLog = 20;
          let liveEnabled = false;
          let audioCtx = null;
          let eventSource = null;
          let nextPlayTime = 0;

          function addLogEntry(data) {
            const mic0 = data.mic0 || {};
            const mic1 = data.mic1 || {};
            const mic2 = data.mic2 || {};
            const ts = mic0.timestamp || mic1.timestamp || mic2.timestamp || '-';
            const entry = `[${ts}] MIC0: ${(mic0.rms || 0).toFixed(0)} | MIC1: ${(mic1.rms || 0).toFixed(0)} | MIC2: ${(mic2.rms || 0).toFixed(0)}`;
            logEntries.unshift(entry);
            if (logEntries.length > maxLog) logEntries.pop();
            
            const logDiv = document.getElementById('log');
            if (logDiv) {
              logDiv.innerHTML = logEntries.map(e => `<div class="log-entry">${e}</div>`).join('');
            }
          }

          function updateRMS(rms, channels) {
            const rmsDiv = document.getElementById('rms');
            if (!rmsDiv) return;
            
            const labels = ['Left', 'Right', 'Mono'];
            rmsDiv.innerHTML = rms.map((val, i) => {
                const normalized = val / 2147483648*2; // Normalize 32-bit RMS to [-1, 1]
              const percent = Math.min(100, normalized); // Normalize to [0, 100]
              console.log(`RMS Channel ${i}: ${val.toFixed(0)} (normalized: ${normalized.toFixed(3)}, percent: ${percent.toFixed(1)}%)`);
              return `
              <div class="rms-item">
                <div class="rms-label">${labels[i] || 'Ch' + (i+1)}</div>
                <div class="rms-value">${val.toFixed(0)}</div>
                <div class="rms-bar-bg">
                  <div class="rms-bar-fill" style="height: ${percent}%"></div>
                </div>
              </div>
            `}).join('');
          }

          function base64ToBytes(b64) {
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
              bytes[i] = binary.charCodeAt(i);
            }
            return bytes;
          }

          function scheduleAudio(packet) {
            if (!audioCtx) return;
            const bytes = base64ToBytes(packet.data || '');
            if (bytes.length === 0) return;

            const channels = packet.channels || 1;  // May be 1 (single UDP packet) or 3 (assembled)
            const bits = packet.bits || 32;  // ESP32 sends 32-bit
            
            // Support 16-bit and 32-bit
            let samples;
            if (bits === 32) {
              samples = new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
            } else {
              samples = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
            }
            
            const frames = Math.floor(samples.length / channels);
            if (frames === 0) {
              console.warn('[Audio] Empty packet received');
              return;
            }

            const sampleRate = packet.sample_rate || 48000;  // ESP32 uses 48kHz
            const buffer = audioCtx.createBuffer(channels, frames, sampleRate);

            // Process samples: normalize to [-1, 1]
            if (channels === 1) {
              const channelData = buffer.getChannelData(0);
              for (let i = 0; i < frames; i++) {
                const sample = samples[i];
                const normalized = bits === 32 
                  ? Math.max(-1, Math.min(1, sample / 2147483648))
                  : sample / 32768;
                channelData[i] = normalized;
              }
            } else if (channels === 3) {
              // 3-channel interleaved: [L, R, C, L, R, C, ...]
              for (let ch = 0; ch < 3; ch++) {
                const channelData = buffer.getChannelData(ch);
                for (let i = 0; i < frames; i++) {
                  const sample = samples[i * 3 + ch];
                  const normalized = bits === 32 
                    ? Math.max(-1, Math.min(1, sample / 2147483648))
                    : sample / 32768;
                  channelData[i] = normalized;
                }
              }
            } else {
              // Generic multi-channel handling
              for (let ch = 0; ch < channels; ch++) {
                const channelData = buffer.getChannelData(ch);
                for (let i = 0; i < frames; i++) {
                  const sample = samples[i * channels + ch];
                  const normalized = bits === 32 
                    ? Math.max(-1, Math.min(1, sample / 2147483648))
                    : sample / 32768;
                  channelData[i] = normalized;
                }
              }
            }

            const source = audioCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(audioCtx.destination);

            // Add jitter buffer: schedule playback with 50ms lookahead to smooth packet arrival jitter
            const jitterBufferMs = 50;  // 50ms smoothing buffer
            const minScheduleTime = audioCtx.currentTime + (jitterBufferMs / 1000);
            
            if (nextPlayTime < minScheduleTime) {
              nextPlayTime = minScheduleTime;
            }
            
            source.start(nextPlayTime);
            nextPlayTime += buffer.duration;
          }

          function startLive() {
            if (liveEnabled) return;
            liveEnabled = true;
            document.getElementById('liveBtn').textContent = 'Stop Listening';

            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            nextPlayTime = audioCtx.currentTime + 0.1;

            const listenMode = document.getElementById('listenMode').value;
            const streamUrl = listenMode === '3ch' ? '/api/stream_3ch' : '/api/stream';
            const selectedMic = listenMode.startsWith('mic') ? parseInt(listenMode.substring(3)) : -1;

            eventSource = new EventSource(streamUrl);
            eventSource.onmessage = (event) => {
              try {
                const packet = JSON.parse(event.data);
                
                // If in single-channel mode, filter to specific mic
                if (selectedMic >= 0) {
                  // Only process packets from the selected mic
                  const device = packet.device_id;
                  if (!device.includes(`mic${selectedMic}`)) {
                    return;
                  }
                }
                
                scheduleAudio(packet);
              } catch (err) {
                console.error('Stream parse error:', err);
              }
            };
            eventSource.onerror = (err) => {
              console.error('Stream error:', err);
            };
          }

          async function stopLive() {
            if (!liveEnabled) return;
            liveEnabled = false;
            document.getElementById('liveBtn').textContent = 'Listen Live';

            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
            if (audioCtx) {
              await audioCtx.close();
              audioCtx = null;
            }
            nextPlayTime = 0;
          }

          async function refresh() {
            try {
              const resp = await fetch('/api/latest');
              if (!resp.ok) {
                document.getElementById('status').textContent = 'Error: ' + resp.status;
                return;
              }
              const data = await resp.json();
              
              console.log('Received data:', data);
              
              if (!data || (!data.mic0 && !data.mic1 && !data.mic2)) {
                document.getElementById('status').textContent = 'Waiting for data...';
                return;
              }
              
              packetCount++;
              
              // Display info for all three mics
              const activeMics = [];
              if (data.mic0 && data.mic0.timestamp) activeMics.push('MIC0');
              if (data.mic1 && data.mic1.timestamp) activeMics.push('MIC1');
              if (data.mic2 && data.mic2.timestamp) activeMics.push('MIC2');
              
              document.getElementById('status').textContent = '✓ Receiving: ' + activeMics.join(', ') + ' (polls: ' + packetCount + ')';
              
              // Display combined info (use mic0 as primary, or first available)
              const primaryMic = data.mic0 || data.mic1 || data.mic2;
              document.getElementById('device').textContent = activeMics.join(', ');
              document.getElementById('rate').textContent = (primaryMic.sample_rate || '-') + ' Hz';
              document.getElementById('channels').textContent = '3 (1 per mic)';
              document.getElementById('bits').textContent = (primaryMic.bits || '-') + ' bit';
              document.getElementById('bytes').textContent = [
                data.mic0 ? data.mic0.byte_count : 0,
                data.mic1 ? data.mic1.byte_count : 0,
                data.mic2 ? data.mic2.byte_count : 0
              ].join(' / ');
              document.getElementById('ts').textContent = primaryMic.timestamp || '-';
              document.getElementById('count').textContent = packetCount;
              
              // Update RMS with all three mics
              const rmsValues = [
                data.mic0 ? data.mic0.rms : 0,
                data.mic1 ? data.mic1.rms : 0,
                data.mic2 ? data.mic2.rms : 0
              ];
              updateRMS(rmsValues, 3);
              addLogEntry(data);
            } catch (e) {
              console.error('Fetch error:', e);
              document.getElementById('status').textContent = 'Error: ' + e.message;
            }
          }

          document.getElementById('saveBtn').addEventListener('click', async () => {
            try {
              const resp = await fetch('/api/download_buffer');
              if (!resp.ok) {
                alert('No audio data available');
                return;
              }
              const blob = await resp.blob();
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = 'audio_' + new Date().toISOString().replace(/[:.]/g, '-') + '.wav';
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              window.URL.revokeObjectURL(url);
            } catch (e) {
              console.error('Download error:', e);
              alert('Download failed: ' + e.message);
            }
          });

          document.getElementById('liveBtn').addEventListener('click', async () => {
            if (liveEnabled) {
              await stopLive();
            } else {
              startLive();
            }
          });

          document.getElementById('textBtn').addEventListener('click', async () => {
            try {
              const resp = await fetch('/api/download_buffer_text');
              if (!resp.ok) {
                alert('No audio data available');
                return;
              }
              const blob = await resp.blob();
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = 'audio_' + new Date().toISOString().replace(/[:.]/g, '-') + '.csv';
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              window.URL.revokeObjectURL(url);
            } catch (e) {
              console.error('Download error:', e);
              alert('Download failed: ' + e.message);
            }
          });

          document.getElementById('diagBtn').addEventListener('click', async () => {
            try {
              const resp = await fetch('/api/diagnostics');
              if (!resp.ok) {
                alert('Failed to fetch diagnostics');
                return;
              }
              const data = await resp.json();
              console.log('Diagnostics:', data);
              
              document.getElementById('total-packets').textContent = 
                data.packet_counters.total + ' (MIC0: ' + data.packet_counters.mic0 + 
                ', MIC1: ' + data.packet_counters.mic1 + 
                ', MIC2: ' + data.packet_counters.mic2 + ')';
              document.getElementById('total-errors').textContent = 
                data.packet_errors.total + ' (MIC0: ' + data.packet_errors.mic0 + 
                ', MIC1: ' + data.packet_errors.mic1 + 
                ', MIC2: ' + data.packet_errors.mic2 + ')';
              document.getElementById('mic0-packets').textContent = data.packet_counters.mic0;
              document.getElementById('mic1-packets').textContent = data.packet_counters.mic1;
              document.getElementById('mic2-packets').textContent = data.packet_counters.mic2;
              document.getElementById('last-mic0').textContent = data.latest_packets.mic0.last_update || 'Never';
              document.getElementById('last-mic1').textContent = data.latest_packets.mic1.last_update || 'Never';
              document.getElementById('last-mic2').textContent = data.latest_packets.mic2.last_update || 'Never';
            } catch (e) {
              console.error('Diagnostics error:', e);
            }
          });

          setInterval(refresh, 40);
          refresh();
        </script>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.route("/api/upload", methods=["POST", "OPTIONS"])
def api_upload() -> Response:
    if request.method == "OPTIONS":
        return Response("", status=200)

    data = request.get_data(cache=False) or b""
    packet = {
        "device_id": request.headers.get("X-Device-Id"),
        "sample_rate": _parse_int(request.headers.get("X-Sample-Rate")),
        "channels": _parse_int(request.headers.get("X-Channels")) or 1,
        "bits": _parse_int(request.headers.get("X-Bits")) or 16,
        "format": request.headers.get("X-Format") or "pcm16le",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": data,
    }

    with latest_lock:
        # For backward compatibility with HTTP POST, store in mic0
        latest_mic0.update({
            "device_id": packet["device_id"] or "http-upload",
            "sample_rate": packet["sample_rate"],
            "channels": packet["channels"],
            "bits": packet["bits"],
            "timestamp": packet["timestamp"],
            "data": packet["data"],
        })

    # Add to circular buffer
    with audio_buffer_lock:
        # Store raw bytes; we'll handle WAV formatting on download
        audio_buffer.append(data)
        global buffer_sample_rate, buffer_channels, buffer_bits
        buffer_sample_rate = packet["sample_rate"] or 48000
        buffer_channels = packet["channels"] or 3
        buffer_bits = packet["bits"] or 32

    for queue in list(subscribers):
        queue.put(packet)

    return Response("OK", status=200)


@app.route("/api/latest")
def api_latest() -> Response:
    with latest_lock:
        mic0_snapshot = dict(latest_mic0)
        mic1_snapshot = dict(latest_mic1)
        mic2_snapshot = dict(latest_mic2)

    # Return data for all three mics
    return jsonify(
        {
            "mic0": {
                "device_id": mic0_snapshot.get("device_id"),
                "sample_rate": mic0_snapshot.get("sample_rate"),
                "channels": mic0_snapshot.get("channels"),
                "bits": mic0_snapshot.get("bits"),
                "timestamp": mic0_snapshot.get("timestamp"),
                "byte_count": len(mic0_snapshot.get("data") or b""),
                "rms": mic0_snapshot.get("rms", 0.0),
            },
            "mic1": {
                "device_id": mic1_snapshot.get("device_id"),
                "sample_rate": mic1_snapshot.get("sample_rate"),
                "channels": mic1_snapshot.get("channels"),
                "bits": mic1_snapshot.get("bits"),
                "timestamp": mic1_snapshot.get("timestamp"),
                "byte_count": len(mic1_snapshot.get("data") or b""),
                "rms": mic1_snapshot.get("rms", 0.0),
            },
            "mic2": {
                "device_id": mic2_snapshot.get("device_id"),
                "sample_rate": mic2_snapshot.get("sample_rate"),
                "channels": mic2_snapshot.get("channels"),
                "bits": mic2_snapshot.get("bits"),
                "timestamp": mic2_snapshot.get("timestamp"),
                "byte_count": len(mic2_snapshot.get("data") or b""),
                "rms": mic2_snapshot.get("rms", 0.0),
            },
        }
    )


def reassemble_3channel_audio():
    """Reassemble 3-channel interleaved audio from per-mic buffers.
    
    Returns tuple (interleaved_pcm_bytes, num_frames) where interleaved_pcm_bytes
    contains 3-channel 32-bit PCM interleaved as: [mic0_ch0, mic1_ch0, mic2_ch0, mic0_ch1, ...]
    """
    with audio_buffer_lock:
        # Get snapshot of all 3 mic buffers
        mic0_packets = list(audio_buffer_mic0)
        mic1_packets = list(audio_buffer_mic1)
        mic2_packets = list(audio_buffer_mic2)
    
    if not mic0_packets and not mic1_packets and not mic2_packets:
        return b"", 0
    
    # Concatenate all packets per channel to get continuous PCM data
    mic0_data = b"".join(mic0_packets) if mic0_packets else b""
    mic1_data = b"".join(mic1_packets) if mic1_packets else b""
    mic2_data = b"".join(mic2_packets) if mic2_packets else b""
    
    # Convert bytes to int32 samples
    mic0_samples = np.frombuffer(mic0_data, dtype=np.int32) if mic0_data else np.array([], dtype=np.int32)
    mic1_samples = np.frombuffer(mic1_data, dtype=np.int32) if mic1_data else np.array([], dtype=np.int32)
    mic2_samples = np.frombuffer(mic2_data, dtype=np.int32) if mic2_data else np.array([], dtype=np.int32)
    
    # Ensure all channels have same length (pad with zeros if needed)
    max_len = max(len(mic0_samples), len(mic1_samples), len(mic2_samples))
    if max_len == 0:
        return b"", 0
    
    mic0_samples = np.pad(mic0_samples, (0, max_len - len(mic0_samples)), mode='constant')
    mic1_samples = np.pad(mic1_samples, (0, max_len - len(mic1_samples)), mode='constant')
    mic2_samples = np.pad(mic2_samples, (0, max_len - len(mic2_samples)), mode='constant')
    
    # Interleave: create output array with [mic0, mic1, mic2, mic0, mic1, mic2, ...]
    interleaved = np.zeros(max_len * 3, dtype=np.int32)
    interleaved[0::3] = mic0_samples
    interleaved[1::3] = mic1_samples
    interleaved[2::3] = mic2_samples
    
    # Convert back to bytes
    interleaved_bytes = interleaved.astype(np.int32).tobytes()
    
    return interleaved_bytes, max_len


@app.route("/api/process_latest", methods=["POST"])
def api_process_latest() -> Response:
    """
    Run transcription + localization on the most recent audio in the 3 mic buffers.
    Body (optional JSON):
      {
        "max_packets": 40,
        "convert_to_int16": true
      }
    """
    body = request.get_json(silent=True) or {}
    max_packets = int(body.get("max_packets", 40))
    convert_to_int16 = bool(body.get("convert_to_int16", True))

    chunks = build_latest_3ch_chunks_from_buffers(
        max_packets=max_packets,
        convert_to_int16=convert_to_int16,
    )

    if not chunks:
        return jsonify({"ok": False, "error": "No aligned audio available in buffers"}), 404

    try:
        # IMPORTANT: if you convert to int16 above, tell downstream bits=16 if you pass it
        # (depends on how your engine_kwargs are used inside TranscriptionLocalizationSession).
        segments = audio_main.run(
            chunks=chunks,
            num_channels=3,
            # Example engine kwargs you might need (only if your stack uses them):
            # sample_rate=48000,
            # sample_width=2 if convert_to_int16 else 4,
        )
        return jsonify({"ok": True, "segments": segments, "num_chunks": len(chunks)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _int32_3ch_to_int16_bytes(interleaved_i32: np.ndarray) -> bytes:
    """
    Convert interleaved int32 samples to int16.
    Common for pipelines that assume 16-bit PCM.
    """
    # Scale: take the top 16 bits (fast) and clip to int16 range.
    i16 = np.clip((interleaved_i32 >> 16), -32768, 32767).astype(np.int16)
    return i16.tobytes()


def build_latest_3ch_chunks_from_buffers(
    *,
    max_packets: int = 40,
    convert_to_int16: bool = True,
) -> List[bytes]:
    """
    Build multi-channel chunks suitable for audio_main.run(chunks, num_channels=3).

    Each chunk corresponds to one 'packet time slice' assembled from mic0/mic1/mic2.
    """
    with audio_buffer_lock:
        b0 = list(audio_buffer_mic0)
        b1 = list(audio_buffer_mic1)
        b2 = list(audio_buffer_mic2)

    n = min(len(b0), len(b1), len(b2))
    if n == 0:
        return []

    # take the most recent aligned packets
    n_take = min(n, max_packets)
    b0 = b0[-n_take:]
    b1 = b1[-n_take:]
    b2 = b2[-n_take:]

    chunks: List[bytes] = []

    for p0, p1, p2 in zip(b0, b1, b2):
        s0 = np.frombuffer(p0, dtype=np.int32)
        s1 = np.frombuffer(p1, dtype=np.int32)
        s2 = np.frombuffer(p2, dtype=np.int32)

        # Keep packets aligned by trimming to the shortest packet length
        m = min(len(s0), len(s1), len(s2))
        if m == 0:
            continue
        s0 = s0[:m]
        s1 = s1[:m]
        s2 = s2[:m]

        interleaved = np.empty(m * 3, dtype=np.int32)
        interleaved[0::3] = s0
        interleaved[1::3] = s1
        interleaved[2::3] = s2

        if convert_to_int16:
            chunks.append(_int32_3ch_to_int16_bytes(interleaved))
        else:
            chunks.append(interleaved.tobytes())

    return chunks



@app.route("/api/download_buffer")
def api_download_buffer() -> Response:
    """Download buffered audio as WAV file.
    
    Returns 3-channel 32-bit PCM at 48kHz with interleaved channel data:
    [mic0_sample0, mic1_sample0, mic2_sample0, mic0_sample1, ...]
    """
    # Reassemble 3-channel interleaved audio from per-mic buffers
    interleaved_bytes, num_frames = reassemble_3channel_audio()
    
    if num_frames == 0:
        return Response("No audio data buffered", status=404)
    
    # Create WAV file in memory
    wav_io = io.BytesIO()
    
    try:
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(3)           # 3-channel mic array (mic0, mic1, mic2)
            wav_file.setsampwidth(4)            # 32-bit = 4 bytes per sample
            wav_file.setframerate(48000)        # ESP32 uses 48kHz
            wav_file.writeframes(interleaved_bytes)
        
        wav_io.seek(0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}_{buffer_bits}bit.wav"
        
        return send_file(
            wav_io,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Error creating WAV: {e}")
        return Response(f"Error creating WAV: {e}", status=500)


@app.route("/api/download_buffer_text")
def api_download_buffer_text() -> Response:
    with audio_buffer_lock:
        if len(audio_buffer) == 0:
            return Response("No audio data buffered", status=404)
        
        # Concatenate all buffered chunks
        combined = b"".join(audio_buffer)
        
        # Parse as signed integers based on bit depth
        bytes_per_sample = buffer_bits // 8
        num_samples = len(combined) // bytes_per_sample
        
        csv_lines = ["sample_index,value"]
        for i in range(num_samples):
            start = i * bytes_per_sample
            end = start + bytes_per_sample
            sample = int.from_bytes(combined[start:end], "little", signed=True)
            csv_lines.append(f"{i},{sample}")
        
        csv_text = "\n".join(csv_lines)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}_{buffer_bits}bit.csv"
        
        return Response(
            csv_text,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )




@app.route("/api/diagnostics")
def api_diagnostics() -> Response:
    """Return packet reception diagnostics for all three mics."""
    with latest_lock:
        mic0_ts = latest_mic0.get("timestamp")
        mic1_ts = latest_mic1.get("timestamp")
        mic2_ts = latest_mic2.get("timestamp")
    
    return jsonify({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "packet_counters": {
            "mic0": packet_counters[0],
            "mic1": packet_counters[1],
            "mic2": packet_counters[2],
            "total": sum(packet_counters.values()),
        },
        "packet_errors": {
            "mic0": packet_errors[0],
            "mic1": packet_errors[1],
            "mic2": packet_errors[2],
            "total": sum(packet_errors.values()),
        },
        "latest_packets": {
            "mic0": {"last_update": mic0_ts, "size": len(latest_mic0.get("data", "")) if latest_mic0.get("data") else 0},
            "mic1": {"last_update": mic1_ts, "size": len(latest_mic1.get("data", "")) if latest_mic1.get("data") else 0},
            "mic2": {"last_update": mic2_ts, "size": len(latest_mic2.get("data", "")) if latest_mic2.get("data") else 0},
        },
        "buffer_size": len(audio_buffer),
        "subscribers": len(subscribers),
    })


@app.route("/api/stream")
def api_stream() -> Response:
    queue: Queue = Queue()
    subscribers.append(queue)

    def gen():
        try:
            while True:
                packet = queue.get()
                data = packet.get("data") or ""
                # Data should already be base64 encoded from UDP receiver
                if isinstance(data, bytes):
                    data = base64.b64encode(data).decode("ascii")
                payload = {
                    "device_id": packet.get("device_id"),
                    "sample_rate": packet.get("sample_rate"),
                    "channels": packet.get("channels"),
                    "bits": packet.get("bits"),
                    "format": packet.get("format"),
                    "timestamp": packet.get("timestamp"),
                    "data": data,
                }
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            if queue in subscribers:
                subscribers.remove(queue)

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/stream_3ch")
def api_stream_3ch() -> Response:
    """Stream 3-channel interleaved audio assembled from per-mic buffers."""
    
    def gen():
        last_mic0_len = 0
        last_mic1_len = 0
        last_mic2_len = 0
        chunk_size = 256  # Frames to send per packet
        
        try:
            while True:
                with audio_buffer_lock:
                    # Get current buffer sizes
                    mic0_len = len(audio_buffer_mic0)
                    mic1_len = len(audio_buffer_mic1)
                    mic2_len = len(audio_buffer_mic2)
                    
                    # Only send if all buffers have new data
                    if mic0_len > last_mic0_len and mic1_len > last_mic1_len and mic2_len > last_mic2_len:
                        # Get new packets from each buffer
                        mic0_packets = list(audio_buffer_mic0)[last_mic0_len:mic0_len]
                        mic1_packets = list(audio_buffer_mic1)[last_mic1_len:mic1_len]
                        mic2_packets = list(audio_buffer_mic2)[last_mic2_len:mic2_len]
                        
                        # Concatenate packets to get continuous PCM data
                        mic0_data = b"".join(mic0_packets)
                        mic1_data = b"".join(mic1_packets)
                        mic2_data = b"".join(mic2_packets)
                        
                        # Update tracking
                        last_mic0_len = mic0_len
                        last_mic1_len = mic1_len
                        last_mic2_len = mic2_len
                        
                        # Convert to samples
                        mic0_samples = np.frombuffer(mic0_data, dtype=np.int32) if mic0_data else np.array([], dtype=np.int32)
                        mic1_samples = np.frombuffer(mic1_data, dtype=np.int32) if mic1_data else np.array([], dtype=np.int32)
                        mic2_samples = np.frombuffer(mic2_data, dtype=np.int32) if mic2_data else np.array([], dtype=np.int32)
                        
                        # Pad to same length
                        max_len = max(len(mic0_samples), len(mic1_samples), len(mic2_samples))
                        if max_len > 0:
                            mic0_samples = np.pad(mic0_samples, (0, max_len - len(mic0_samples)), mode='constant')
                            mic1_samples = np.pad(mic1_samples, (0, max_len - len(mic1_samples)), mode='constant')
                            mic2_samples = np.pad(mic2_samples, (0, max_len - len(mic2_samples)), mode='constant')
                            
                            # Send in chunks
                            for start_idx in range(0, max_len, chunk_size):
                                end_idx = min(start_idx + chunk_size, max_len)
                                chunk_samples = end_idx - start_idx
                                
                                # Interleave 3 channels: [m0, m1, m2, m0, m1, m2, ...]
                                interleaved = np.zeros(chunk_samples * 3, dtype=np.int32)
                                interleaved[0::3] = mic0_samples[start_idx:end_idx]
                                interleaved[1::3] = mic1_samples[start_idx:end_idx]
                                interleaved[2::3] = mic2_samples[start_idx:end_idx]
                                
                                # Encode as base64
                                chunk_bytes = interleaved.astype(np.int32).tobytes()
                                data_b64 = base64.b64encode(chunk_bytes).decode('utf-8')
                                
                                payload = {
                                    "device_id": "3ch-assembled",
                                    "sample_rate": 48000,
                                    "channels": 3,
                                    "bits": 32,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "data": data_b64,
                                }
                                yield f"data: {json.dumps(payload)}\n\n"
                
                time.sleep(0.016)  # ~60fps update rate
        except GeneratorExit:
            pass
    
    return Response(gen(), mimetype="text/event-stream")

def _generate_audio_data(frames: int, channels: int, phase: float = 0.0) -> bytes:
    """Generate sine wave test audio data at 48kHz 16-bit"""
    frequencies = [440.0, 554.37, 659.25]  # A4, C#5, E5
    samples = []
    for i in range(frames):
        for ch in range(channels):
            freq = frequencies[ch % len(frequencies)]
            t = (i + phase) / 48000.0
            amplitude = 8000 + 4000 * math.sin(2 * math.pi * 0.5 * t)  # varying amplitude
            value = int(amplitude * math.sin(2 * math.pi * freq * t))
            samples.append(max(-32768, min(32767, value)))
    return struct.pack(f"<{len(samples)}h", *samples)


def simulate_post(frames: int = 256, sample_rate: int = 48000, channels: int = 3, phase: float = 0.0) -> Dict[str, object]:
    bytes_per_sample = 2
    data = _generate_audio_data(frames, channels, phase)
    headers = {
        "X-Device-Id": "test-device",
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": str(channels),
        "X-Bits": str(bytes_per_sample * 8),
        "X-Format": "pcm16le",
        "Content-Type": "application/octet-stream",
    }

    with app.test_client() as client:
        resp = client.post("/api/upload", data=data, headers=headers)

    return {
        "status_code": resp.status_code,
        "byte_count": len(data),
        "device_id": headers["X-Device-Id"],
    }


def simulate_post_remote(url: str, frames: int = 256, sample_rate: int = 48000, channels: int = 3, phase: float = 0.0) -> Dict[str, object]:
    bytes_per_sample = 2
    data = _generate_audio_data(frames, channels, phase)
    headers = {
        "X-Device-Id": "test-device",
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": str(channels),
        "X-Bits": str(bytes_per_sample * 8),
        "X-Format": "pcm16le",
        "Content-Type": "application/octet-stream",
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        status = resp.status

    return {
        "status_code": status,
        "byte_count": len(data),
        "device_id": headers["X-Device-Id"],
        "url": url,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AudioVision server")
    parser.add_argument("--simulate", action="store_true", help="send a test upload to /api/upload")
    parser.add_argument("--remote", default="", help="remote upload URL for simulation")
    parser.add_argument("--loop", action="store_true", help="continuously send data (use with --simulate)")
    parser.add_argument("--interval", type=float, default=0.016, help="seconds between packets in loop mode (default: 16ms)")
    parser.add_argument("--frames", type=int, default=256, help="frames to send when simulating")
    parser.add_argument("--rate", type=int, default=48000, help="sample rate for simulation")
    parser.add_argument("--channels", type=int, default=3, help="channels for simulation")
    parser.add_argument("--host", default="0.0.0.0", help="host to bind the server")
    parser.add_argument("--port", type=int, default=30000, help="port to bind the server")
    parser.add_argument("--debug", action="store_true", help="enable Flask debug")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.simulate:
        phase = 0.0
        count = 0
        try:
            while True:
                if args.remote:
                    result = simulate_post_remote(args.remote, frames=args.frames, sample_rate=args.rate, channels=args.channels, phase=phase)
                else:
                    result = simulate_post(frames=args.frames, sample_rate=args.rate, channels=args.channels, phase=phase)
                count += 1
                phase += args.frames
                print(f"\r[{count}] {json.dumps(result)}", end="", flush=True)
                if not args.loop:
                    print()
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n\nSent {count} packets")
    else:
        # Start UDP receiver threads with delay to allow system to clean up old sockets
        print("Starting UDP receivers...")
        
        # Check if ports are available
        print("Checking port availability...")
        ports_to_check = [(UDP_PORT_MIC0, 0), (UDP_PORT_MIC1, 1), (UDP_PORT_MIC2, 2)]
        for port, mic_id in ports_to_check:
            if _is_port_available(port):
                print(f"✓ Port {port} (MIC{mic_id}) available")
            else:
                print(f"✗ Port {port} (MIC{mic_id}) IN USE - will retry with SO_REUSEADDR")
        
        print("Giving system 2 seconds to clean up old sockets...")
        time.sleep(2)
        
        for mic_id, port in enumerate([UDP_PORT_MIC0, UDP_PORT_MIC1, UDP_PORT_MIC2]):
            thread = threading.Thread(target=udp_receiver, args=(port, mic_id), daemon=True)
            print(f"Starting UDP receiver for MIC{mic_id} on port {port}...")
            thread.start()
            udp_threads.append(thread)
            time.sleep(0.5)  # Stagger thread starts
        
        print(f"UDP receivers active on ports {UDP_PORT_MIC0}, {UDP_PORT_MIC1}, {UDP_PORT_MIC2}")
        app.run(host=args.host, port=args.port, debug=args.debug)
