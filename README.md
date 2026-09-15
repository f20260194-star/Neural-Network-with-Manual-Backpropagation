# Neural-Network-with-Manual-Backpropagation

 Neural Network From Scratch (NumPy)

A minimal feedforward neural network implemented using only NumPy — forward
pass, manually-derived backpropagation, gradient checking, and training —
with no autograd/deep learning framework used inside the core
implementation. PyTorch is used only as a *reference* to verify gradients
are correct (see `tests/test_gradients.py`).

## Architecture

```
Input (64) -> Linear(64,32) -> ReLU -> Linear(32,10) -> Softmax -> Cross-Entropy Loss
```

- Input: 8x8 grayscale digit images (sklearn `load_digits`), flattened to 64 features
- Hidden layer: 32 units, ReLU activation
- Output: 10 classes (digits 0-9), softmax + cross-entropy loss

```

## Setup

Requires Python 3.8+.

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `numpy`, `scikit-learn` (for the digits dataset),
`matplotlib` (for the loss curve plot), and `torch` (used ONLY as a
reference oracle for gradient checking — not used anywhere in the core
network implementation).

## How to run

**Train the network:**
```bash
python src/train.py
```
This trains the network on the digits dataset and prints the training loss
at regular intervals, and saves a loss-curve plot to `loss_curve.png`.

**Run the correctness harness (gradient check):**
```bash
python tests/test_gradients.py
```
This constructs a small random network and input batch, computes gradients
three ways — (1) manually via the hand-derived backward pass, (2) via
`torch.autograd`, (3) via numerical (finite-difference) gradient checking —
and reports a PASS/FAIL based on whether the manual gradients match the
other two within a tolerance (see WRITEUP.md for the tolerance used and the
actual results).

## Results

- Gradient check: all four parameters (`W1`, `b1`, `W2`, `b2`) match
  numerical (finite-difference) gradients to within `1e-9`–`1e-11` max
  relative error. 

See WRITEUP.md for full derivations, results, and discussion.
