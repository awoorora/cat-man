# cat-man

Constant-velocity **Kalman filter baseline** — adaptive trajectory prediction. This is the "simple fixed-model baseline" for the learned model to be benchmarked against.

Companion to the data pipeline (`cat-apult`) and the model repo (`cat-walk`); consumed by `cat-alogue`, which runs this filter and the LSTM through identical evaluation code.

---

## What it is

A 6-state Kalman filter — position and velocity in 3D, `[x, y, z, vx, vy, vz]` — with a **constant-velocity** motion model and a position-only measurement. It denoises a window of noisy position fixes into a smoothed state, then propagates that state forward to predict future positions.

Being a constant-velocity model, it is near-optimal on straight-line motion and degrades on turns / acceleration — which is exactly the gap a learned model is meant to close. That contrast is the point of having it as the baseline.

Single file, pure `numpy`, no dependencies beyond that.

---

## Interface

Implements the shared predictor contract, so it is interchangeable with the LSTM in the evaluation harness:

```python
from catman import KalmanBaseline

kf = KalmanBaseline(sigma_meas=1.0, sigma_acc=3.0)

future   = kf.predict(past_pos, dt, horizons)      # (W,3) noisy in -> (len(horizons),3) out
pos, vel = kf.estimate_state(past_pos, dt)         # denoised state at the last fix

# batched variants for evaluating a whole test set in one call:
future   = kf.predict_batch(past_pos, dt, horizons)        # (N,W,3) -> (N,len(horizons),3)
pos, vel = kf.estimate_state_batch(past_pos, dt)           # -> (N,3), (N,3)
```

### Tuning knobs

| Parameter | Default | Meaning |
| --- | --- | --- |
| `sigma_meas` | `1.0` | measurement-noise std (m). Set it to the actual sensor noise so the filter trusts measurements appropriately — the eval harness sets this to the sweep's σ at each point, making it a fairly-tuned baseline. |
| `sigma_acc` | `3.0` | process noise: how much acceleration the constant-velocity model tolerates before it stops trusting its own prediction. |

`predict` propagates the smoothed state forward `dt` per step and samples the requested horizons. `estimate_state` returns the denoised position and velocity at the last input fix (`k` controls how many recent fixes it filters over).

---

## Layout

```
catman.py     # KalmanBaseline (+ Catman / Kitten aliases)
```

> `Catman` and `Kitten` are cosmetic aliases for `KalmanBaseline`.
