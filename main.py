import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SAMPLE_RATE   = 1000        # samples per second
DURATION      = 2.0         # seconds
FREQ_HZ       = 30          # dominant vibration frequency

# Thresholds
WARN_THRESHOLD = 3.0
CRIT_THRESHOLD = 6.0

# Signal parameters — tweak these to change health state
NOISE_LEVEL   = 2.5         # background noise amplitude
FAULT_STRENGTH = 3.5        # periodic fault spike amplitude (0= no fault)
FAULT_INTERVAL = 0.15       # seconds between fault spikes


def generate_vibration(sample_rate, duration, freq, noise, fault_amp, fault_int):
    
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Base sinusoidal component
    signal = np.sin(2 * np.pi * freq * t) * noise * 0.4

    # Random broadband noise
    signal += np.random.randn(len(t)) * noise * 0.6

    # Periodic fault spikes
    if fault_amp > 0:
        spike_idx = (t % fault_int < 1 / sample_rate)
        signal += spike_idx * fault_amp * np.sign(np.random.randn(len(t)))

    return t, signal


def compute_rms(signal):
    return float(np.sqrt(np.mean(signal ** 2)))


def classify(rms):
    if rms >= CRIT_THRESHOLD:
        return "CRITICAL", "#E24B4A"
    elif rms >= WARN_THRESHOLD:
        return "WARNING", "#BA7517"
    else:
        return "NORMAL", "#1D9E75"


def plot(t, signal, rms, status, color):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Machine Health Monitor", fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.plot(t, signal, color=color, linewidth=0.8, alpha=0.9)

    # Warning and critical threshold bands
    ax.axhspan( WARN_THRESHOLD,  CRIT_THRESHOLD, alpha=0.08, color="#BA7517", label="Warning zone")
    ax.axhspan(-CRIT_THRESHOLD, -WARN_THRESHOLD, alpha=0.08, color="#BA7517")
    ax.axhspan( CRIT_THRESHOLD,  signal.max() + 1, alpha=0.08, color="#E24B4A", label="Critical zone")
    ax.axhspan( signal.min() - 1, -CRIT_THRESHOLD, alpha=0.08, color="#E24B4A")

    ax.axhline( WARN_THRESHOLD, color="#BA7517", linewidth=0.8, linestyle="--")
    ax.axhline(-WARN_THRESHOLD, color="#BA7517", linewidth=0.8, linestyle="--")
    ax.axhline( CRIT_THRESHOLD, color="#E24B4A", linewidth=0.8, linestyle="--")
    ax.axhline(-CRIT_THRESHOLD, color="#E24B4A", linewidth=0.8, linestyle="--")

    ax.set_ylabel("Amplitude (mm/s)", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title(f"Vibration Waveform  |  RMS = {rms:.3f} mm/s", fontsize=12)
    ax.grid(True, alpha=0.25)

    warn_patch = mpatches.Patch(color="#BA7517", alpha=0.3, label=f"Warning ≥ {WARN_THRESHOLD}")
    crit_patch = mpatches.Patch(color="#E24B4A", alpha=0.3, label=f"Critical ≥ {CRIT_THRESHOLD}")
    ax.legend(handles=[warn_patch, crit_patch], fontsize=9, loc="upper right")

    # Status bar 
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Background pill
    pill = mpatches.FancyBboxPatch((0.02, 0.1), 0.96, 0.8,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, alpha=0.15,
                                   edgecolor=color, linewidth=1.5)
    ax2.add_patch(pill)

    ax2.text(0.5, 0.55, f"STATUS: {status}",
             ha="center", va="center", fontsize=18, fontweight="bold",
             color=color, transform=ax2.transAxes)

    hint = {
        "NORMAL":   "Machine is operating within safe limits.",
        "WARNING":  "Elevated vibration detected. Schedule maintenance soon.",
        "CRITICAL": "Dangerous vibration level! Stop machine and inspect immediately."
    }[status]
    ax2.text(0.5, 0.22, hint,
             ha="center", va="center", fontsize=10, color=color,
             alpha=0.85, transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig("vibration_report.png", dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Plot saved → vibration_report.png")
    plt.show()


def main():
    print("=" * 50)
    print("  Vibration Health Monitoring System")
    print("=" * 50)

    t, signal = generate_vibration(
        SAMPLE_RATE, DURATION, FREQ_HZ,
        NOISE_LEVEL, FAULT_STRENGTH, FAULT_INTERVAL
    )

    rms = compute_rms(signal)
    peak = float(np.max(np.abs(signal)))
    status, color = classify(rms)

    print(f"\n  Samples   : {len(signal)}")
    print(f"  Duration  : {DURATION} s  @  {SAMPLE_RATE} Hz")
    print(f"  RMS       : {rms:.4f} mm/s")
    print(f"  Peak      : {peak:.4f} mm/s")
    print(f"\n  *** {status} ***")
    print("=" * 50)

    plot(t, signal, rms, status, color)


if __name__ == "__main__":
    main()