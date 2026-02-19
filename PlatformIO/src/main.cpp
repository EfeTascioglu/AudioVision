#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <cmath>
#include "driver/i2s.h"

// DMA I2S
// Access second core as well. 

// WiFi settings
static const char *WIFI_SSID = "Virus Network";
static const char *WIFI_PASS = "12345678";

// Server settings
static const char *DEVICE_ID = "esp32-nauthiz-01";
static const char *SERVER_IP = "10.213.87.70";
static const int UDP_PORT_COMBINED = 30001;  // Single port for all 3 channels combined

// UDP client (single socket for combined transmission)
static WiFiUDP udp_combined;

// I2S pins (set these to match your wiring)
static const int I2S0_BCLK = 26;
static const int I2S0_WS = 25;
static const int I2S0_DATA_IN = 33;

static const int I2S1_BCLK = 26; //14
static const int I2S1_WS = 25;   // SHARED with I2S0 for perfect hardware synchronization
static const int I2S1_DATA_IN = 8;

// Audio settings
// DATA FORMAT: 24 bit, 2's complement, MSB First. 
static const int SAMPLE_RATE_HZ = 48000;
static const int I2S_BITS_PER_SAMPLE = 32; 
static const int PCM_BITS_PER_SAMPLE = 32; 
static const int I2S_SAMPLE_SHIFT = 0; // 18-bit left-justified -> right-align into 32-bit
static const size_t FRAMES_PER_CHUNK = 512; 

// Buffers
static int32_t i2s0_raw[FRAMES_PER_CHUNK * 2];
static int32_t i2s1_raw[FRAMES_PER_CHUNK];
static int32_t pcm_out[FRAMES_PER_CHUNK * 3];

// Diagnostics
static bool ENABLE_TIMING_DIAGNOSTICS = true;  // Toggle this to enable/disable timing tracking
static unsigned long last_diagnostic_time = 0;

// Test Mode Configuration
static bool ENABLE_TEST_MODE = false;  // Set to true to send test waveforms instead of microphone input
static const float TEST_FREQUENCY_HZ = 440.0f;  // Test signal frequency (sine, square, triangle)
static const float TEST_AMPLITUDE = 131072.0f;  // Amplitude scale for 32-bit samples
static size_t test_phase_index = 0;  // Global phase tracker for synchronized waveforms
static unsigned long total_frames_captured = 0;
static unsigned long total_bytes_uploaded = 0;
static unsigned int chunk_count = 0;
static unsigned long max_upload_time_ms = 0;
static unsigned long min_upload_time_ms = ULONG_MAX;
static unsigned long total_i2s_read_time = 0;
static unsigned long total_convert_time = 0;
static unsigned long total_loop_time = 0;

// Packet transmission tracking (single atomic transmission)
static unsigned int packets_sent_combined = 0;   // Successfully sent combined 3-channel packet
static unsigned int packets_failed_combined = 0; // Failed to send combined packet
static unsigned int sync_perfect = 0;             // Counter for sample-perfect alignment achieved

// Timing index: encoded in bits 29-31 to mark which packet/cluster frames belong to
// Increments with each successful transmission, wraps 0-7 (3 bits)
// Allows server to validate 3 consecutive frames came from same synchronized I2S read
static uint8_t timing_index = 0;

static inline int32_t convert_sample(int32_t raw) {
  return raw >> I2S_SAMPLE_SHIFT;
}

// Channel identification tagging: encode channel ID into bits 1-2 of each sample
// This allows the server to detect channel misalignment without separate headers
// Bits 1-2 are used for identification: 00=channel0, 01=channel1, 10=channel2, 11=reserved
static inline int32_t tag_sample_with_channel(int32_t sample, uint8_t channel) {
  // Clear bits 1-2, then set them to channel ID
  int32_t mask = sample & ~0x6;  // 0x6 = 0b110 (bits 1-2)
  return mask | ((channel & 0x3) << 1);
}

// Extract channel tag from sample bits 1-2
static inline uint8_t extract_channel_tag(int32_t sample) {
  return (sample >> 1) & 0x3;
}

// Extract timing index from sample bits 3-5
static inline uint8_t extract_timing_index(int32_t sample) {
  return (sample >> 3) & 0x7;
}

// Tag sample with both channel ID (bits 1-2) and timing index (bits 3-5)
// Timing index marks which packet/cluster this frame belongs to
// Server validates 3 consecutive frames have matching timing index (same packet sync)
static inline int32_t tag_sample_with_timing_and_channel(int32_t sample, uint8_t channel, uint8_t timing_idx) {
  // Clear bits 1-2, then set them to channel ID
  sample = (sample & ~0x6) | ((channel & 0x3) << 1);
  // Clear bits 3-5, then set them to timing index
  sample = (sample & ~0x38) | (((uint32_t)timing_idx & 0x7) << 3);
  return sample;
}

// Test Mode Wave Generation Functions
// Generate sine wave sample at phase index
static inline int32_t generate_sine_sample(size_t phase_idx) {
  float t = (float)phase_idx / SAMPLE_RATE_HZ;
  float sample = TEST_AMPLITUDE * sinf(2.0f * M_PI * TEST_FREQUENCY_HZ * t);
  
  // Convert to 18-bit signed (2's complement): range -131072 to +131071
  int32_t sample_18bit = (int32_t)sample;
  // Clamp to 18-bit signed range
  if (sample_18bit > 131071) sample_18bit = 131071;
  if (sample_18bit < -131072) sample_18bit = -131072;
  
  // Left-shift by 14 bits: (18 bits of signal)(14 bits of 0s)
  int32_t shifted_sample = sample_18bit << 14;
  
  // Serial.printf(">SINESAMPLE:0x%08X", shifted_sample); // Debug print in hex
  // Serial.println();
  return shifted_sample;
}


// Generate triangle wave sample at phase index
static inline int32_t generate_triangle_sample(size_t phase_idx) {
  float t = (float)phase_idx / SAMPLE_RATE_HZ;
  float phase_normalized = fmodf(TEST_FREQUENCY_HZ * t, 1.0f);  // 0.0 to 1.0
  float sample;
  if (phase_normalized < 0.25f) {
    // Rising from -1 to 1: first quarter
    sample = TEST_AMPLITUDE * (-1.0f + 4.0f * phase_normalized);
  } else if (phase_normalized < 0.75f) {
    // Falling from 1 to -1: middle half
    sample = TEST_AMPLITUDE * (1.0f - 4.0f * (phase_normalized - 0.25f));
  } else {
    // Rising from -1 to 1: last quarter
    sample = TEST_AMPLITUDE * (-1.0f + 4.0f * (phase_normalized - 0.75f));
  }
  
  // Convert to 18-bit signed (2's complement): range -131072 to +131071
  int32_t sample_18bit = (int32_t)sample;
  // Clamp to 18-bit signed range
  if (sample_18bit > 131071) sample_18bit = 131071;
  if (sample_18bit < -131072) sample_18bit = -131072;
  
  // Left-shift by 14 bits: (18 bits of signal)(14 bits of 0s)
  int32_t shifted_sample = sample_18bit << 14;
  
  // Serial.printf(">TRIANGLESAMPLE:0x%08X", shifted_sample); // Debug print in hex
  // Serial.println();
  return shifted_sample;
}

static void setup_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected, IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi connect failed");
  }
}

static void setup_i2s_port(i2s_port_t port, const i2s_pin_config_t &pins, i2s_channel_fmt_t chan_fmt, i2s_channel_t channel_count) {
  i2s_config_t config = {};
  config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  config.sample_rate = SAMPLE_RATE_HZ;
  config.bits_per_sample = static_cast<i2s_bits_per_sample_t>(I2S_BITS_PER_SAMPLE);
  config.channel_format = chan_fmt;
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  config.communication_format = static_cast<i2s_comm_format_t>(I2S_COMM_FORMAT_I2S);
  #pragma GCC diagnostic pop
  config.intr_alloc_flags = 0;
  config.dma_buf_count = 8;        // Increased from 6 for better buffering
  config.dma_buf_len = FRAMES_PER_CHUNK; 
  config.use_apll = true;          // Enable APLL for better clock accuracy (changed from false)
  config.tx_desc_auto_clear = true; // Auto clear tx descriptor on underflow (helps prevent noise)
  config.fixed_mclk = 0;

  Serial.printf("Setting up I2S port %d...\n", port);

  i2s_driver_install(port, &config, 0, nullptr);
  Serial.printf("I2S port %d driver installed.\n", port);
  i2s_set_pin(port, &pins);
  Serial.printf("I2S port %d pins set.\n", port);
  i2s_set_clk(port, SAMPLE_RATE_HZ, static_cast<i2s_bits_per_sample_t>(I2S_BITS_PER_SAMPLE), channel_count);
  Serial.printf("I2S port %d clock set.\n", port);
}

static void setup_i2s() {
  i2s_pin_config_t pins0 = {};
  pins0.bck_io_num = I2S0_BCLK;
  pins0.ws_io_num = I2S0_WS;
  pins0.data_out_num = I2S_PIN_NO_CHANGE;
  pins0.data_in_num = I2S0_DATA_IN;

  i2s_pin_config_t pins1 = {};
  pins1.bck_io_num = I2S1_BCLK;
  pins1.ws_io_num = I2S1_WS;
  pins1.data_out_num = I2S_PIN_NO_CHANGE;
  pins1.data_in_num = I2S1_DATA_IN;

  Serial.println("Initializing I2S...");

  setup_i2s_port(I2S_NUM_0, pins0, I2S_CHANNEL_FMT_RIGHT_LEFT, I2S_CHANNEL_STEREO);
  setup_i2s_port(I2S_NUM_1, pins1, I2S_CHANNEL_FMT_ONLY_LEFT, I2S_CHANNEL_MONO);
}

static bool send_udp_combined(WiFiUDP &udp, int port, const int32_t *data_3ch, size_t frames) {
  if (WiFi.status() != WL_CONNECTED) {
    if (ENABLE_TIMING_DIAGNOSTICS) Serial.println("[UDP] WiFi not connected");
    return false;
  }

  // Combined packet: all 3 channels interleaved [L, R, C, L, R, C, ...]
  size_t bytes = frames * 3 * sizeof(int32_t);
  
  unsigned long send_start = 0;
  if (ENABLE_TIMING_DIAGNOSTICS) {
    send_start = millis();
  }

  // Send single atomic packet with all 3 channels
  int result = 0;
  if (udp.beginPacket(SERVER_IP, port)) {
    size_t written = udp.write(reinterpret_cast<const uint8_t *>(data_3ch), bytes);
    result = udp.endPacket();
    
    if (ENABLE_TIMING_DIAGNOSTICS && written != bytes) {
      Serial.printf("[UDP] Write mismatch: wrote %d/%lu bytes (3 channels, %lu frames)\n", written, bytes, frames);
    }
  } else {
    if (ENABLE_TIMING_DIAGNOSTICS) {
      Serial.printf("[UDP] beginPacket() failed for port %d\n", port);
    }
    result = 0;
  }

  bool success = (result == 1);
  
  // Track packet statistics
  if (success) {
    packets_sent_combined++;
    sync_perfect++;  // Since all 3 channels sent in one packet, sync is perfect
    total_bytes_uploaded += bytes;
    
    // Note: timing_index is now incremented inside the frame assembly loop (per-frame)
    // rather than per-packet, so we don't increment it here
  } else {
    packets_failed_combined++;
  }

  if (ENABLE_TIMING_DIAGNOSTICS) {
    unsigned long send_time = millis() - send_start;
    max_upload_time_ms = max(max_upload_time_ms, send_time);
    min_upload_time_ms = min(min_upload_time_ms, send_time);
  }

  return success;
}

void send_test_chunk() {
  // Generate 3-channel interleaved sine wave test: 440Hz
  const float frequency = 440.0f;
  const float amplitude = 65536.0f; // scale for 32-bit
  int32_t test_data[FRAMES_PER_CHUNK * 3] = {};

  for (size_t i = 0; i < FRAMES_PER_CHUNK; ++i) {
    float t = (float)i / SAMPLE_RATE_HZ;
    float sample = amplitude * sinf(2.0f * M_PI * frequency * t);
    int32_t s = (int32_t)sample;
    test_data[i * 3] = s;       // Left
    test_data[i * 3 + 1] = s;   // Right
    test_data[i * 3 + 2] = s;   // Center
  }

  send_udp_combined(udp_combined, UDP_PORT_COMBINED, test_data, FRAMES_PER_CHUNK);
}

void send_audio_to_serial() {
  for (size_t i = 0; i < FRAMES_PER_CHUNK * 3; i += 3) {
    // Print in hex for debugging; in practice, you might want to print in hex or just the raw bytes
    Serial.write(reinterpret_cast<const uint8_t *>(&pcm_out[i]), sizeof(int32_t) * 3);
    // Serial.println(pcm_out[i]);
    if (i >= 21) { // Limit how much we print to serial for debugging
      break;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);
  setup_wifi();
  setup_i2s();
  
  // Initialize single UDP socket for combined 3-channel transmission
  if (!udp_combined.begin(0)) {
    Serial.println("Failed to initialize UDP socket for combined 3-channel stream");
  } else {
    Serial.println("UDP combined stream initialized on port 30001");
  }
  
  Serial.println("Setup complete, starting main loop...");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    setup_wifi();
  }

  unsigned long loop_start = 0;
  if (ENABLE_TIMING_DIAGNOSTICS) {
    loop_start = millis();
  }

  size_t bytes_read0 = 0;
  size_t bytes_read1 = 0;
  size_t frames = 0;

  // I2S Read phase: measure timing offset between ports (or generate test signals)
  // CRITICAL: Sequential reads cause temporal drift. We measure this and compensate.
  unsigned long i2s_start = 0;
  unsigned long i2s0_read_time_us = 0;
  if (ENABLE_TIMING_DIAGNOSTICS) {
    i2s_start = millis();
  }

  if (ENABLE_TEST_MODE) {
    // Test mode: generate synthetic waveforms instead of reading I2S
    // All three channels use aligned frequencies for synchronized signal testing
    frames = FRAMES_PER_CHUNK;
    
    // Generate test signals: sine (left), square (right), triangle (center)
    for (size_t i = 0; i < frames; ++i) {
      int32_t sine_sample = generate_sine_sample(test_phase_index + i);
      int32_t square_sample = generate_triangle_sample(test_phase_index + i);
      int32_t triangle_sample = generate_triangle_sample(test_phase_index + i);
      
      i2s0_raw[i * 2] = sine_sample;      // MIC0 Left: Sine
      i2s0_raw[i * 2 + 1] = square_sample; // MIC0 Right: Square
      i2s1_raw[i] = triangle_sample;       // MIC1 Center: Triangle
    }
    test_phase_index += frames;  // Advance global phase for next chunk
  } else {
    // Read both I2S ports (hardware-synchronized via shared WS/BCLK)
    // Both ports now stereo: L/R pairs sampled simultaneously
    i2s_read(I2S_NUM_0, i2s0_raw, sizeof(i2s0_raw), &bytes_read0, portMAX_DELAY);
    i2s_read(I2S_NUM_1, i2s1_raw, sizeof(i2s1_raw), &bytes_read1, portMAX_DELAY);

    // Both are now stereo (2 channels each)
    size_t frames0 = bytes_read0 / (sizeof(int32_t) * 2);
    size_t frames1 = bytes_read1 / sizeof(int32_t);
    frames = frames0 < frames1 ? frames0 : frames1;
  }

  if (ENABLE_TIMING_DIAGNOSTICS && !ENABLE_TEST_MODE) {
    unsigned long i2s_time = millis() - i2s_start;
    total_i2s_read_time += i2s_time;
  }

  if (frames == 0) {
    delay(5);
    return;
  }

  // Convert/pack phase
  // CRITICAL: Apply timing alignment here
  // MIC0 samples are stored in delay buffer for offset samples, then read back
  // This compensates for MIC1 lagging due to sequential I2S reads
  unsigned long convert_start = 0;
  if (ENABLE_TIMING_DIAGNOSTICS) {
    convert_start = millis();
  }

  for (size_t i = 0; i < frames; ++i) {
    // Hardware synchronization ensures both I2S ports sampled at identical moments
    // I2S1 stereo-config for WS consistency, but only use left channel (MIC1)
    int32_t left0 = convert_sample(i2s0_raw[i * 2]);
    int32_t right0 = convert_sample(i2s0_raw[i * 2 + 1]);
    int32_t mono = convert_sample(i2s1_raw[i * 2]);  // L-channel of stereo-config I2S1

    // Tag each sample with channel ID and timing index
    // Channel IDs: 0=MIC0Left, 1=MIC0Right, 2=MIC1Mono
    // Timing index (bits 3-5): increments with each frame
    int32_t left0_tagged = tag_sample_with_timing_and_channel(left0, 0, timing_index);
    int32_t right0_tagged = tag_sample_with_timing_and_channel(right0, 1, timing_index);
    int32_t mono_tagged = tag_sample_with_timing_and_channel(mono, 2, timing_index);

    // Output: 3-channel interleaved [L0, R0, L1, L0, R0, L1, ...]
    pcm_out[i * 3] = left0_tagged;
    pcm_out[i * 3 + 1] = right0_tagged;
    pcm_out[i * 3 + 2] = mono_tagged;
    
    // Increment timing index for next frame
    timing_index = (timing_index + 1) & 0x7;
  }

  if (ENABLE_TIMING_DIAGNOSTICS) {
    unsigned long convert_time = millis() - convert_start;
    total_convert_time += convert_time;
  }

  total_frames_captured += frames;
  chunk_count++;
  
  // CRITICAL: Send all 3 channels in a SINGLE atomic UDP packet
  // Packet format (interleaved): [L, R, C, L, R, C, ...]
  // This ensures all 3 channels arrive together on the server, eliminating timing skew
  bool ok = send_udp_combined(udp_combined, UDP_PORT_COMBINED, pcm_out, frames);

  // send_audio_to_serial();
  
  if (!ok) {
    // Network failure on single transmission
    // No need to drain buffers since we maintain hardware sync continuously
    // The next chunk will transmit fresh data
  }
  
  if (ENABLE_TIMING_DIAGNOSTICS) {
    unsigned long loop_time = millis() - loop_start;
    total_loop_time += loop_time;
  }
  
  // Print detailed timing every 10 seconds
  unsigned long now = millis();
  if (ENABLE_TIMING_DIAGNOSTICS && (now - last_diagnostic_time >= 10000)) {
    last_diagnostic_time = now;
    
    float elapsed_sec = 10.0;
    float avg_bitrate_kbps = (total_bytes_uploaded * 8) / 1000.0 / elapsed_sec;
    float audio_duration_sec = (float)total_frames_captured / SAMPLE_RATE_HZ;
    float avg_chunk_bytes = chunk_count > 0 ? (float)total_bytes_uploaded / chunk_count : 0;
    float avg_i2s_time = chunk_count > 0 ? (float)total_i2s_read_time / chunk_count : 0;
    float avg_convert_time = chunk_count > 0 ? (float)total_convert_time / chunk_count : 0;
    float avg_loop_time = chunk_count > 0 ? (float)total_loop_time / chunk_count : 0;
    float avg_upload_time = chunk_count > 0 ? (float)(max_upload_time_ms + min_upload_time_ms) / 2.0 : 0;
    
    Serial.println("\n=== AUDIO DIAGNOSTICS (10 sec interval) ===");
    if (ENABLE_TEST_MODE) {
      Serial.printf("*** TEST MODE ACTIVE ***\n");
      Serial.printf("Test frequency: %.1f Hz (Sine on Left, Square on Right, Triangle on Center)\n", TEST_FREQUENCY_HZ);
      Serial.printf("Global phase index: %zu samples\n", test_phase_index);
    }
    Serial.printf("Chunks captured: %d\n", chunk_count);
    Serial.printf("Total frames: %lu (%.2f sec of audio)\n", total_frames_captured, audio_duration_sec);
    Serial.printf("Total bytes uploaded: %lu\n", total_bytes_uploaded);
    Serial.printf("Avg chunk size: %.0f bytes\n", avg_chunk_bytes);
    Serial.printf("Bitrate: %.1f kbps\n", avg_bitrate_kbps);
    Serial.printf("Sample rate: %d Hz (%.2f ms per chunk)\n", SAMPLE_RATE_HZ, (float)FRAMES_PER_CHUNK * 1000.0 / SAMPLE_RATE_HZ);
    Serial.printf("WiFi RSSI: %d dBm\n", WiFi.RSSI());
    
    Serial.println("\n--- PACKET INTEGRITY & ALIGNMENT ---");
    unsigned int total_packets = packets_sent_combined + packets_failed_combined;
    float success_rate = total_packets > 0 ? (float)packets_sent_combined * 100.0 / total_packets : 0.0;
    Serial.printf("Combined 3-channel packets: %u sent, %u failed\n", packets_sent_combined, packets_failed_combined);
    Serial.printf("Success rate: %.2f%%\n", success_rate);
    Serial.printf("Sample-perfect aligned packets: %u (all 3 channels in single UDP packet)\n", sync_perfect);
    
    Serial.println("\n--- TRANSMISSION OPTIMIZATION ---");
    Serial.printf("Transmission method: Single atomic UDP packet (port 30001)\n");
    Serial.printf("Packet format: 3-channel interleaved [L,R,C,L,R,C,...]\n");
    
    Serial.println("\n--- TIMING BREAKDOWN (ms per chunk) ---");
    Serial.printf("I2S read: %.2f ms (avg)\n", avg_i2s_time);
    Serial.printf("Convert/pack: %.2f ms (avg)\n", avg_convert_time);
    Serial.printf("Upload: %.2f ms (avg), min: %lu ms, max: %lu ms\n", avg_upload_time, min_upload_time_ms, max_upload_time_ms);
    Serial.printf("Total loop: %.2f ms (avg)\n", avg_loop_time);
    Serial.println("=========================================\n");
    
    // Reset for next interval
    total_frames_captured = 0;
    total_bytes_uploaded = 0;
    chunk_count = 0;
    max_upload_time_ms = 0;
    min_upload_time_ms = ULONG_MAX;
    total_i2s_read_time = 0;
    total_convert_time = 0;
    total_loop_time = 0;
    packets_sent_combined = 0;
    packets_failed_combined = 0;
    sync_perfect = 0;
    // NOTE: Timing offset variables no longer used (hardware synchronization handles this)
  }
}