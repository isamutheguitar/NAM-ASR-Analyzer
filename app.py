import os
import streamlit as st
import numpy as np
from scipy import signal
import sounddevice as sd
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# ----------------
# App Configuration & Aesthetics
# ----------------
st.set_page_config(page_title="ASR Analyzer", page_icon="🎛️", layout="wide")

st.markdown("""
<style>
.metric-container {
    background: linear-gradient(135deg, #1f1c2c 0%, #928DAB 100%);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    text-align: center;
    color: white;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-value {
    font-size: 4.5rem;
    font-weight: 900;
    color: #fff;
    text-shadow: 0 0 25px rgba(255,255,255,0.6);
    line-height: 1.2;
    font-family: 'Inter', sans-serif;
}
.metric-label {
    font-size: 1.3rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    opacity: 0.9;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------
# Database Setup
# ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "asr_results.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  device_name TEXT,
                  preset TEXT,
                  sample_rate INTEGER,
                  asr_db REAL)''')
    conn.commit()
    conn.close()

def save_result(device_name, preset, sample_rate, asr_db):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT INTO results (device_name, preset, sample_rate, asr_db)
                 VALUES (?, ?, ?, ?)''', (device_name, preset, sample_rate, float(asr_db)))
    conn.commit()
    conn.close()

def load_results():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame(columns=["timestamp", "device_name", "preset", "sample_rate", "asr_db"])
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT timestamp, device_name, preset, sample_rate, asr_db FROM results ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# ----------------
# DSP Functions
# ----------------
def generate_log_sweep(duration, fs, f_start=20.0, f_end=None):
    if f_end is None:
        f_end = fs / 2.0
    t = np.arange(int(duration * fs)) / fs
    K = duration / np.log(f_end / f_start)
    phase = 2 * np.pi * f_start * K * (np.exp(t / K) - 1.0)
    sweep = np.sin(phase)
    
    # Smooth fade in/out to prevent clicks
    fade_len = int(0.01 * fs) # 10ms
    if fade_len > 0 and fade_len * 2 < len(sweep):
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        sweep[:fade_len] *= fade_in
        sweep[-fade_len:] *= fade_out
        
    return sweep, t

def calculate_asr(sweep, recorded, fs, f_start=20.0, f_end=None):
    if f_end is None:
        f_end = fs / 2.0
        
    # 1. Alignment (Latency compensation)
    corr = signal.correlate(recorded, sweep, mode='full', method='fft')
    delay = np.argmax(corr) - len(sweep) + 1
    
    if np.max(np.abs(corr)) < 1e-5:
        # Prevent math errors if empty signal
        return 0.0, np.array([0]), np.array([0]), np.zeros((1,1)), np.array([0]), np.array([0])
        
    if delay > 0:
        aligned = recorded[delay:]
    elif delay < 0:
        aligned = np.pad(recorded, (-delay, 0))
    else:
        aligned = recorded
        
    # Match length exactly
    if len(aligned) > len(sweep):
        aligned = aligned[:len(sweep)]
    else:
        aligned = np.pad(aligned, (0, len(sweep) - len(aligned)))
        
    # 2. STFT with Blackman-Harris window
    nperseg = 4096
    f, t_stft, Zxx = signal.stft(aligned, fs=fs, window='blackmanharris', nperseg=nperseg, noverlap=nperseg*3//4)
    Sxx = np.abs(Zxx)**2
    
    E_signal_total = 0.0
    E_alias_total = 0.0
    E_signal_history = []
    E_alias_history = []
    
    duration = len(sweep) / fs
    df = fs / nperseg
    
    # 3. Energy Calculation
    for i, t_val in enumerate(t_stft):
        t_eff = min(max(t_val, 0.0), duration)
        # Instantaneous Fundamental Frequency based on sweep time
        inst_f = f_start * (f_end / f_start)**(t_eff / duration)
        
        harmonic_mask = np.zeros(len(f), dtype=bool)
        h = 1
        # Track fundamental and positive integer harmonics up to Nyquist
        while h * inst_f < fs / 2.0:
            hf = h * inst_f
            # Proportional Notch Width: +/- 15% of current frequency, min 50Hz
            notch_width_hz = max(50.0, hf * 0.15)
            notch_bins = max(1, int(notch_width_hz / df))
            
            bin_idx = int(np.round(hf / df))
            start_bin = max(0, bin_idx - notch_bins)
            end_bin = min(len(f), bin_idx + notch_bins + 1)
            harmonic_mask[start_bin:end_bin] = True
            h += 1
            
        # Ignore DC and extreme low freq noise for signal calculation
        valid_signal_bins = f >= 20.0
        # Ignore frequencies below 500Hz for alias calculation to prevent low-frequency leakage from skewing the ASR
        valid_alias_bins = f >= 500.0
        
        alias_mask = valid_alias_bins & (~harmonic_mask)
        
        frame_energy = Sxx[:, i]
        e_sig = np.sum(frame_energy[harmonic_mask & valid_signal_bins])
        e_ali = np.sum(frame_energy[alias_mask])
        
        E_signal_total += e_sig
        E_alias_total += e_ali
        E_signal_history.append(e_sig)
        E_alias_history.append(e_ali)
        
    asr = 10 * np.log10(E_alias_total / E_signal_total) if E_signal_total > 0 else 0.0
    Sxx_db = 10 * np.log10(Sxx + 1e-12) # For Spectrogram Visualization
    
    # Energy History in dB
    sig_db_history = 10 * np.log10(np.array(E_signal_history) + 1e-12)
    ali_db_history = 10 * np.log10(np.array(E_alias_history) + 1e-12)
    
    return asr, f, t_stft, Sxx_db, sig_db_history, ali_db_history

# ----------------
# Main Streamlit App
# ----------------
def main():
    if 'asr_score' not in st.session_state:
        st.session_state.asr_score = None
    if 'spectrogram_data' not in st.session_state:
        st.session_state.spectrogram_data = None
    if 'fs' not in st.session_state:
        st.session_state.fs = 48000
        
    init_db()
    
    # Sidebar
    st.sidebar.title("🎛️ Hardware Routing")
    
    if st.sidebar.button("🔄 デバイスリストを更新 (Refresh)", use_container_width=True):
        sd._terminate()
        sd._initialize()
        st.rerun()
        
    try:
        devices = sd.query_devices()
        in_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        out_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_output_channels'] > 0]
    except Exception as e:
        st.sidebar.error(f"Error querying devices: {e}")
        in_devices = []
        out_devices = []
        
    if not in_devices or not out_devices:
        st.error("No valid audio input/output devices found. Please check your system audio settings.")
        st.stop()
        
    in_dev = st.sidebar.selectbox("Audio Input Device", in_devices, format_func=lambda x: f"[{x[0]}] {x[1]}")
    out_dev = st.sidebar.selectbox("Audio Output Device", out_devices, format_func=lambda x: f"[{x[0]}] {x[1]}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        in_ch = st.number_input("Input Ch. (1-based)", min_value=1, value=1)
    with col2:
        out_ch = st.number_input("Output Ch. (1-based)", min_value=1, value=1)
        
    fs_options = [44100, 48000, 88200, 96000, 192000]
    fs = st.sidebar.selectbox("Sample Rate (Hz)", fs_options, index=1)
    
    duration_options = [5.0, 10.0, 20.0, 30.0, 60.0]
    sweep_duration = st.sidebar.selectbox("Sweep Duration (sec)", duration_options, index=2)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 *ASIO drivers are strongly recommended on Windows to bypass OS mixers.*")

    # Main Area
    st.title("ASR Analyzer")
    st.markdown("Measure and visualize digital aliasing noise introduced by non-linear external hardware effectors using a Log Sine Sweep. The lower the ASR score (more negative), the better.")
    
    if st.button("🎙️ Measure ASR", use_container_width=True, type="primary"):
        try:
            max_in = sd.query_devices(in_dev[0])['max_input_channels']
            max_out = sd.query_devices(out_dev[0])['max_output_channels']
            
            if in_ch > max_in or out_ch > max_out:
                st.error(f"Selected channels exceed device capabilities (Max IN: {max_in}, Max OUT: {max_out}).")
            else:
                sweep, t_sweep = generate_log_sweep(duration=sweep_duration, fs=fs, f_start=20.0)
                out_data = np.zeros((len(sweep), max_out), dtype=np.float32)
                out_data[:, out_ch - 1] = sweep
                
                import time
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                # Non-blocking recording
                recorded = sd.playrec(out_data, samplerate=fs, channels=max_in, device=(in_dev[0], out_dev[0]), blocking=False)
                
                total_duration = len(sweep) / fs
                start_time = time.time()
                
                # Countdown loop
                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= total_duration:
                        break
                        
                    progress = min(1.0, elapsed / total_duration)
                    remain = max(0, int(np.ceil(total_duration - elapsed)))
                    
                    progress_text.markdown(f"**🎙️ 測定中... 残り時間: {remain} 秒**")
                    progress_bar.progress(progress)
                    time.sleep(0.1)
                    
                sd.wait() # Ensure audio is fully processed
                progress_text.empty()
                progress_bar.empty()
                
                rec_mono = recorded[:, in_ch - 1]
                
                with st.spinner("🧮 計算・描画中... (Calculating & Drawing...)"):
                    asr, f_stft, t_stft, Sxx_db, sig_db, ali_db = calculate_asr(sweep, rec_mono, fs)
                    
                    # Generate figure inside the spinner
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                    fig.patch.set_facecolor('#0E1117')
                    
                    f_start = 20.0
                    f_end = fs / 2.0
                    f_inst = f_start * (f_end / f_start)**(t_stft / sweep_duration)
                    
                    # --- Upper Plot: Spectrogram ---
                    ax1.set_facecolor('#0E1117')
                    im = ax1.pcolormesh(f_inst, f_stft, Sxx_db, shading='gouraud', cmap='inferno', vmin=np.max(Sxx_db)-80, vmax=np.max(Sxx_db))
                    ax1.set_ylabel('Frequency [Hz]', color='white')
                    ax1.set_yscale('log')
                    ax1.set_ylim([20, f_end])
                    ax1.tick_params(colors='white', which='both')
                    
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    div1 = make_axes_locatable(ax1)
                    cax1 = div1.append_axes("right", size="2%", pad=0.1)
                    cb = fig.colorbar(im, cax=cax1, format="%+2.0f dB")
                    cb.ax.yaxis.set_tick_params(color='white')
                    cb.outline.set_edgecolor('white')
                    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
                    
                    # --- Lower Plot: Energy History ---
                    ax2.set_facecolor('#0E1117')
                    ax2.plot(f_inst, sig_db, label='Signal Energy (E_signal)', color='#00FFAA', linewidth=2)
                    ax2.plot(f_inst, ali_db, label='Alias Energy (E_alias)', color='#FF4444', linewidth=2)
                    ax2.set_ylabel('Energy [dB]', color='white')
                    ax2.set_xlabel('Sweep Frequency [Hz]', color='white')
                    ax2.set_xscale('log')
                    ax2.set_xlim([20, f_end])
                    ax2.tick_params(colors='white', which='both')
                    ax2.grid(True, color='#333333', linestyle='--')
                    ax2.legend(facecolor='#0E1117', edgecolor='white', labelcolor='white')
                    
                    div2 = make_axes_locatable(ax2)
                    cax2 = div2.append_axes("right", size="2%", pad=0.1)
                    cax2.axis('off')
                    
                    st.session_state.main_fig = fig
                    
                st.session_state.fs = fs
                st.session_state.asr_score = asr
                st.rerun()
                
        except sd.PortAudioError as e:
            st.error(f"**Audio Device Error:** `{e}`\n\nEnsure the selected devices are not exclusively locked by another application.")
        except Exception as e:
            st.error(f"**Unexpected Error:** `{e}`")

    # Results Section
    if st.session_state.asr_score is not None:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Aliasing-to-Signal Ratio</div>
            <div class="metric-value">{st.session_state.asr_score:.2f} dB</div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'main_fig' in st.session_state:
            st.subheader("📊 Spectrogram & Energy History")
            st.pyplot(st.session_state.main_fig, clear_figure=False)

        # Save to DB Area
        with st.expander("💾 Save Result", expanded=True):
            with st.form("save_form"):
                c1, c2 = st.columns(2)
                with c1:
                    device_name = st.text_input("Device Name", "My Hardware Effector")
                with c2:
                    preset_name = st.text_input("Preset / Settings", "Default")
                
                if st.form_submit_button("Save to Database"):
                    save_result(device_name, preset_name, st.session_state.fs, st.session_state.asr_score)
                    st.success("✅ Result saved successfully!")
                    
    st.markdown("---")
    st.subheader("📋 Measurement History")
    df_history = load_results()
    if not df_history.empty:
        df_display = df_history.copy()
        def format_asr(x):
            try:
                if isinstance(x, bytes):
                    try:
                        return f"{np.frombuffer(x, dtype=np.float64)[0]:.2f} dB"
                    except:
                        return f"{float(x.decode('utf-8')):.2f} dB"
                return f"{float(x):.2f} dB"
            except:
                return "N/A"
                
        df_display['asr_db'] = df_display['asr_db'].apply(format_asr)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No measurements saved yet.")

if __name__ == "__main__":
    main()
