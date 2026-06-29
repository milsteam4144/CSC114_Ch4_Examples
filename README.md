# Deep Learning with Python (3rd Edition) — Chapter 4

The three worked examples from Chapter 4 ("Classification and regression"),
split out of the companion notebook into standalone Python scripts that run
cleanly in a GitHub Codespace (or any plain Python environment).

| File | Example | Task |
| ---- | ------- | ---- |
| `example_01_imdb_binary_classification.py` | Classifying movie reviews | Binary classification (IMDb) |
| `example_02_reuters_multiclass_classification.py` | Classifying newswires | Multiclass classification (Reuters) |
| `example_03_california_housing_regression.py` | Predicting house prices | Regression (California Housing) |

## Running in a Codespace

If you open this folder as a Codespace, the included `.devcontainer`
installs everything automatically. Otherwise:

```bash
pip install -r requirements.txt
```

Then run any example:

```bash
python example_01_imdb_binary_classification.py
python example_02_reuters_multiclass_classification.py
python example_03_california_housing_regression.py
```

The datasets download automatically on first run via Keras.

## Notes

- **Plots:** Codespaces are headless, so by default each script *saves* its
  plots as PNG files in the working directory instead of opening a window.
  Pass `--show` to open interactive windows if you're on a machine with a
  display.

- **Backend:** These use Keras 3, which runs on JAX, TensorFlow, or PyTorch.
  The scripts default to JAX (`requirements.txt` installs the CPU build). To
  switch, set the environment variable before running, e.g.:

  ```bash
  KERAS_BACKEND=tensorflow python example_01_imdb_binary_classification.py
  ```

  (You'd also need to `pip install tensorflow` or `pip install torch`.)

- **Speed:** Example 3 trains many models via K-fold validation and can take
  several minutes on CPU. Use `--quick` for a fast smoke test with far fewer
  epochs:

  ```bash
  python example_03_california_housing_regression.py --quick
  ```

These scripts mirror the code from the official book notebooks:
<https://github.com/fchollet/deep-learning-with-python-notebooks>
