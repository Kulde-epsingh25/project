# LSTM Compatibility Note

TensorFlow is currently not available for Python 3.14 on this machine.

Impact:
- ARIMA/SARIMA/Prophet run normally.
- LSTM sections in training output are reported as NaN because TensorFlow cannot be imported.

Options to enable LSTM:
1. Create a Python 3.11 virtual environment and install TensorFlow there.
2. Keep Python 3.14 for statistical models and dashboard only.

Recommended environment for full stack (including LSTM):
- Python 3.10 or 3.11
- TensorFlow 2.x
- Same project code can be reused.
