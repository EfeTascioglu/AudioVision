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
static const int UDP_PORT_MIC0 = 30001;
static const int UDP_PORT_MIC1 = 30002;
static const int UDP_PORT_MIC2 = 30003;

// UDP clients
static WiFiUDP udp_mic0;
static WiFiUDP udp_mic1;
static WiFiUDP udp_mic2;

// I2S pins (set these to match your wiring)
static const int I2S0_BCLK = 26;
static const int I2S0_WS = 25;
static const int I2S0_DATA_IN = 33;

static const int I2S1_BCLK = 14;
static const int I2S1_WS = 27;
static const int I2S1_DATA_IN = 8;

// Audio settings
// DATA FORMAT: 24 bit, 2's complement, MSB First. 
static const int SAMPLE_RATE_HZ = 48000;
static const int I2S_BITS_PER_SAMPLE = 32; // many I2S mics output <=24-bit in a 32-bit frame
static const int PCM_BITS_PER_SAMPLE = 32; // 
static const int I2S_SAMPLE_SHIFT = 0; // 18-bit left-justified -> right-align into 32-bit
static const size_t FRAMES_PER_CHUNK = 512;  // Increased from 256 for better throughput (reduced HTTP overhead)

// Buffers
static int32_t i2s0_raw[FRAMES_PER_CHUNK * 2];
static int32_t i2s1_raw[FRAMES_PER_CHUNK];
static int32_t pcm_out[FRAMES_PER_CHUNK * 3];

// Diagnostics
static bool ENABLE_TIMING_DIAGNOSTICS = true;  // Toggle this to enable/disable timing tracking
static unsigned long last_diagnostic_time = 0;
static unsigned long total_frames_captured = 0;
static unsigned long total_bytes_uploaded = 0;
static unsigned int chunk_count = 0;
static unsigned long max_upload_time_ms = 0;
static unsigned long min_upload_time_ms = ULONG_MAX;
static unsigned long total_i2s_read_time = 0;
static unsigned long total_convert_time = 0;
static unsigned long total_loop_time = 0;

// Packet integrity tracking
static unsigned int packets_sent_mic0 = 0;
static unsigned int packets_sent_mic1 = 0;
static unsigned int packets_sent_mic2 = 0;
static unsigned int packets_failed_mic0 = 0;
static unsigned int packets_failed_mic1 = 0;
static unsigned int packets_failed_mic2 = 0;

static inline int32_t convert_sample(int32_t raw) {
  return raw >> I2S_SAMPLE_SHIFT;
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
  config.communication_format = static_cast<i2s_comm_format_t>(I2S_COMM_FORMAT_I2S);
  config.intr_alloc_flags = 0;
  config.dma_buf_count = 8;        // Increased from 6 for better buffering
  config.dma_buf_len = FRAMES_PER_CHUNK;  // Optimized for 1024-frame chunks
  config.use_apll = true;          // Enable APLL for better clock accuracy (changed from false)
  config.tx_desc_auto_clear = false;
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

static bool send_udp_chunk(WiFiUDP &udp, int port, const int32_t *data, size_t samples) {
  if (WiFi.status() != WL_CONNECTED) {
    if (ENABLE_TIMING_DIAGNOSTICS) Serial.println("[UDP] WiFi not connected");
    return false;
  }

  size_t bytes = samples * sizeof(int32_t);
  
  unsigned long send_start = 0;
  if (ENABLE_TIMING_DIAGNOSTICS) {
    send_start = millis();
  }

  // Send packet
  int result = 0;
  if (udp.beginPacket(SERVER_IP, port)) {
    size_t written = udp.write(reinterpret_cast<const uint8_t *>(data), bytes);
    result = udp.endPacket();
    
    if (ENABLE_TIMING_DIAGNOSTICS && written != bytes) {
      Serial.printf("[UDP] Write mismatch: wrote %d/%lu bytes to port %d\n", written, bytes, port);
    }
  } else {
    if (ENABLE_TIMING_DIAGNOSTICS) {
      Serial.printf("[UDP] beginPacket() failed for port %d\n", port);
    }
    result = 0;
  }

  bool success = (result == 1);
  
  // Track packet statistics per channel
  if (port == UDP_PORT_MIC0) {
    if (success) packets_sent_mic0++; else packets_failed_mic0++;
  } else if (port == UDP_PORT_MIC1) {
    if (success) packets_sent_mic1++; else packets_failed_mic1++;
  } else if (port == UDP_PORT_MIC2) {
    if (success) packets_sent_mic2++; else packets_failed_mic2++;
  }

  if (ENABLE_TIMING_DIAGNOSTICS) {
    unsigned long send_time = millis() - send_start;
    if (success) {
      total_bytes_uploaded += bytes;
    }
    max_upload_time_ms = max(max_upload_time_ms, send_time);
    min_upload_time_ms = min(min_upload_time_ms, send_time);
  } else {
    if (success) {
      total_bytes_uploaded += bytes;
    }
  }

  return success;
}

void send_test_chunk() {
  // Generate single-channel sine wave test: 440Hz
  const float frequency = 440.0f;
  const float amplitude = 65536.0f; // scale for 32-bit
  int32_t test_data[FRAMES_PER_CHUNK] = {};
  
  for (size_t i = 0; i < FRAMES_PER_CHUNK; ++i) {
    float t = (float)i / SAMPLE_RATE_HZ;
    float sample = amplitude * sinf(2.0f * M_PI * frequency * t);
    test_data[i] = (int32_t)sample;  // Channel 0: sine wave
  }
  
  // Send test sine wave to all three channels/ports
  send_udp_chunk(udp_mic0, UDP_PORT_MIC0, test_data, FRAMES_PER_CHUNK);
  delayMicroseconds(100);
  send_udp_chunk(udp_mic1, UDP_PORT_MIC1, test_data, FRAMES_PER_CHUNK);
  delayMicroseconds(100);
  send_udp_chunk(udp_mic2, UDP_PORT_MIC2, test_data, FRAMES_PER_CHUNK);
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
  
  // Initialize UDP sockets with specific source ports to avoid conflicts
  if (!udp_mic0.begin(0)) {
    Serial.println("Failed to initialize UDP socket for MIC0");
  } else {
    Serial.println("UDP MIC0 initialized");
  }
  
  if (!udp_mic1.begin(0)) {
    Serial.println("Failed to initialize UDP socket for MIC1");
  } else {
    Serial.println("UDP MIC1 initialized");
  }
  
  if (!udp_mic2.begin(0)) {
    Serial.println("Failed to initialize UDP socket for MIC2");
  } else {
    Serial.println("UDP MIC2 initialized");
  }
  
  send_test_chunk();
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

  // I2S Read phase
  unsigned long i2s_start = 0;
  if (ENABLE_TIMING_DIAGNOSTICS) {
    i2s_start = millis();
  }

  i2s_read(I2S_NUM_0, i2s0_raw, sizeof(i2s0_raw), &bytes_read0, portMAX_DELAY);
  i2s_read(I2S_NUM_1, i2s1_raw, sizeof(i2s1_raw), &bytes_read1, portMAX_DELAY);

  if (ENABLE_TIMING_DIAGNOSTICS) {
    unsigned long i2s_time = millis() - i2s_start;
    total_i2s_read_time += i2s_time;
  }

  size_t frames0 = bytes_read0 / (sizeof(int32_t) * 2);
  size_t frames1 = bytes_read1 / sizeof(int32_t);
  size_t frames = frames0 < frames1 ? frames0 : frames1;

  if (frames == 0) {
    delay(5);
    return;
  }

  // Convert/pack phase
  unsigned long convert_start = 0;
  if (ENABLE_TIMING_DIAGNOSTICS) {
    convert_start = millis();
  }

  for (size_t i = 0; i < frames; ++i) {
    int32_t left = convert_sample(i2s0_raw[i * 2]);
    int32_t right = convert_sample(i2s0_raw[i * 2 + 1]);
    int32_t mono = convert_sample(i2s1_raw[i]);
    // int32_t right = 0xaabbccdd;  // Only use left channel; silence right and mono
    // int32_t mono = 0xabcd0000;

    pcm_out[i * 3] = left;
    pcm_out[i * 3 + 1] = right;
    pcm_out[i * 3 + 2] = mono;
  }

  if (ENABLE_TIMING_DIAGNOSTICS) {
    unsigned long convert_time = millis() - convert_start;
    total_convert_time += convert_time;
  }

  total_frames_captured += frames;
  chunk_count++;
  
  // Extract channels and send separately via UDP
  int32_t channel0[FRAMES_PER_CHUNK];
  int32_t channel1[FRAMES_PER_CHUNK];
  int32_t channel2[FRAMES_PER_CHUNK];
  
  for (size_t i = 0; i < frames; ++i) {
    channel0[i] = pcm_out[i * 3];
    channel1[i] = pcm_out[i * 3 + 1];
    channel2[i] = pcm_out[i * 3 + 2];
  }
  
  // Send each channel to its corresponding UDP port
  delayMicroseconds(100);
  bool ok0 = send_udp_chunk(udp_mic0, UDP_PORT_MIC0, channel0, frames);
  delayMicroseconds(100);
  bool ok1 = send_udp_chunk(udp_mic1, UDP_PORT_MIC1, channel1, frames);
  delayMicroseconds(100);
  bool ok2 = send_udp_chunk(udp_mic2, UDP_PORT_MIC2, channel2, frames);
  
  // send_audio_to_serial();
  
  bool ok = ok0 && ok1 && ok2;
  if (!ok) {
    delay(20);
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
    Serial.printf("Chunks captured: %d\n", chunk_count);
    Serial.printf("Total frames: %lu (%.2f sec of audio)\n", total_frames_captured, audio_duration_sec);
    Serial.printf("Total bytes uploaded: %lu\n", total_bytes_uploaded);
    Serial.printf("Avg chunk size: %.0f bytes\n", avg_chunk_bytes);
    Serial.printf("Bitrate: %.1f kbps\n", avg_bitrate_kbps);
    Serial.printf("Sample rate: %d Hz (%.2f ms per chunk)\n", SAMPLE_RATE_HZ, (float)FRAMES_PER_CHUNK * 1000.0 / SAMPLE_RATE_HZ);
    Serial.printf("WiFi RSSI: %d dBm\n", WiFi.RSSI());
    
    Serial.println("\n--- PACKET INTEGRITY ---");
    unsigned int total_sent = packets_sent_mic0 + packets_sent_mic1 + packets_sent_mic2;
    unsigned int total_failed = packets_failed_mic0 + packets_failed_mic1 + packets_failed_mic2;
    unsigned int total_packets = total_sent + total_failed;
    float success_rate = total_packets > 0 ? (float)total_sent * 100.0 / total_packets : 0.0;
    Serial.printf("Total packets: %u (sent: %u, failed: %u)\n", total_packets, total_sent, total_failed);
    Serial.printf("Success rate: %.2f%%\n", success_rate);
    Serial.printf("MIC0: %u sent, %u failed (%.1f%%)\n", packets_sent_mic0, packets_failed_mic0, 
                  (packets_sent_mic0 + packets_failed_mic0) > 0 ? (float)packets_sent_mic0 * 100.0 / (packets_sent_mic0 + packets_failed_mic0) : 0.0);
    Serial.printf("MIC1: %u sent, %u failed (%.1f%%)\n", packets_sent_mic1, packets_failed_mic1,
                  (packets_sent_mic1 + packets_failed_mic1) > 0 ? (float)packets_sent_mic1 * 100.0 / (packets_sent_mic1 + packets_failed_mic1) : 0.0);
    Serial.printf("MIC2: %u sent, %u failed (%.1f%%)\n", packets_sent_mic2, packets_failed_mic2,
                  (packets_sent_mic2 + packets_failed_mic2) > 0 ? (float)packets_sent_mic2 * 100.0 / (packets_sent_mic2 + packets_failed_mic2) : 0.0);
    
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
    packets_sent_mic0 = 0;
    packets_sent_mic1 = 0;
    packets_sent_mic2 = 0;
    packets_failed_mic0 = 0;
    packets_failed_mic1 = 0;
    packets_failed_mic2 = 0;
  }
}