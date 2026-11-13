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
Zero-shot accuracy against chance.
Verification: what matched peft, and the parameter counts.
Sweep: accuracy per seed for each method, trainable parameters, percent.
All-token against answer-only loss at rank 16.
Memory: predicted and measured bytes per method.

![Accuracy vs. Trainable Parameters](figures/accuracy_vs_trainable.png)
This figure compares the mean classification accuracy of full fine-tuning and LoRA at ranks 1, 4, and 16. Error bars represent the standard deviation across three seeds.

## Conclusion
What LoRA saved, what it cost, and what these measurements
do and do not establish.

## Known limitations
Anything you are aware of that does not work, or that you would improve
with more time.