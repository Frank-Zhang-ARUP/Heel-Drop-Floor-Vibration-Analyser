# Heel Drop Floor Vibration Analyser

Streamlit app for repeated heel-drop WAV analysis.

## Outputs
- Automatic heel-drop detection
- Event-by-event peak acceleration in m/s², mm/s² and %g
- Free-decay Peak amplitude spectrum
- Repeatable modal-frequency candidates across impacts
- Approximate damping ratio from narrow-band decay envelope
- Excel export

## Deploy on Streamlit Community Cloud
Repository root must contain `app.py` and `requirements.txt`. Set Main file path to `app.py`.

## Important calibration
Absolute acceleration requires the sensor sensitivity, recording-chain gain and WAV 0 dBFS full-scale voltage. Frequency results can still be meaningful without amplitude calibration, but acceleration magnitudes cannot.

## Interpretation
The app reports repeatable free-decay spectral candidates. A candidate is not automatically a proven structural mode. Confirm uncertain results using repeated tests, multiple sensor positions, an instrumented impact hammer/FRF test or measured mode shapes.
