# Fine-Tune with LoRA
A pretrained 135M language model used as a classifier, adapted fully and
with LoRA at several ranks, compared on accuracy, parameters and memory.

## Setup
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Run
Open lab06.ipynb, select the python environment, ensure you have installed all the librearies, and run all the cells from top to bottom
Sections are organized under the headers

## Hardware
GPU: Tesla T4
Total: Time: 4.19 minutes

## Prediction (written before running the sweep)
I expect that LoRA rank 16 will be close to the accuracy of full fine-tuning because it has enough trainable parameters to adapt the model to the task, whereas LoRA rank 1 will be worse because of its smaller update matrices.

## Results
Zero-shot accuracy against chance:
The pretrained model had a zero-shot accuracy of 32.6%, compared to the chance accuracy of 16.67%.

Verification:
The counts match the expected formula summed across the LoRA layers.
The trainable parameter counts were as follows, which match the expected formula summed across the LoRA layers:

| Rank | Trainable Parameters | Percentage |
|---|---:|---:|
| 1 | 57,600 | 0.043% |
| 4 | 230,400 | 0.171% |
| 16 | 921,600 | 0.680% |

Sweep:
| Method | Seed 0 | Seed 1 | Seed 2 | Mean | Learning Rate |
|---|---:|---:|---:|---:|---:|
| Full | 79.0% | 78.8% | 79.6% | 79.1% | 0.00005 |
| LoRA r=1 | 72.2% | 71.6% | 72.2% | 72.0% | 0.001 |
| LoRA r=4 | 76.4% | 74.4% | 74.6% | 75.1% | 0.001 |
| LoRA r=16 | 77.0% | 76.4% | 78.4% | 77.3% | 0.001 |
| LoRA r=16 (all-token) | 71.6% | 70.2% | 71.0% | 70.9% | 0.001 |

Full fine-tuning achieved the highest accuracy, but LoRA rank 16 came close while training less than 1% of the model's parameters. Rank 1 performed the worst of the answer-only methods, as predicted.

![Accuracy vs. Trainable Parameters](figures/accuracy_vs_trainable.png)
This figure compares the mean classification accuracy of full fine-tuning and LoRA at ranks 1, 4, and 16. Error bars represent the standard deviation across three seeds.
As expected in the prediction, rank 16 is closer to the full fine-tuning than it is to rank 4, whereas Rank 1 is clearly the worst.
All-token loss performed worse than answer-only loss at rank 16, decreasing average accuracy from 77.3% to 70.9%.
All fine-tuned methods performed significantly better than the zero-shot baseline of 32.6% and chance accuracy of 16.67%.

Memory:
Memory was predicted using 4 bytes per weight, 4 bytes per trainable gradient, and 8 bytes per trainable parameter for Adam optimizer state.

| Method | Seed 0 (MB) | Seed 1 (MB) | Seed 2 (MB) |
|---|---:|---:|---:|
| Full | 3537.31 | 3533.41 | 3548.44 |
| LoRA r=1 | 1785.50 | 1771.39 | 1785.50 |
| LoRA r=4 | 1787.69 | 1773.59 | 1787.69 |
| LoRA r=16 | 1796.91 | 1782.78 | 1796.91 |
| LoRA r=16 (all-token) | 1796.91 | 1782.78 | 1796.91 |

The predicted and measured memory values matched exactly. LoRA rank 16 reduced training-state memory by approximately 74.3% compared to full fine-tuning.
Peak GPU memory was approximately 3.54 GB for full fine-tuning and 1.79 GB for LoRA rank 16. Peak memory was higher than predicted because training also requires activations and temporary tensors.

## Conclusion
LoRA greatly reduced memory usage and trainable parameters while achieving accuracy close to full fine-tuning. Rank 16 used 74.3% less memory, with only a small accuracy drop from 79.1% to 77.3%. Higher ranks generally performed better, but these results may not apply to other models or datasets.

## Known limitations
The experiments used only one small model, one dataset, three seeds (0, 1, and 2), and a fixed training budget of 120 steps. Results may differ with more training or different models. Training times and peak memory usage also depend on the hardware used.