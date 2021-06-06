# Reference code for the 2021 paper "Audio-Based Fine-Grained Classroom Activity Detection with Neural Networks" 

While we are unable to release pretrained weights or data, we hope for this repository to help resolve any ambiguity found in the paper when reimplementing our experiments.

## Code Flow

Models are defined in [models.py](models.py) and trained with [train.py](train.py). Training will save checkpoints for models at the best F1, mAP, Accuracy and the end of the most recent epoch. Trained model checkpoints are passed to [predict.py](predict.py) which runs the model on in sequential windows over an entire 60-90 minute class session then stiches those together into a single long sequence and pickles the results. The pickled results are fed to [eval.py](eval.py) to generate metrics, PR-curves and confusion matricies, [contributions.py](visualization/contributions.py) to generate Figure 4 from the paper, or [trace.py](visualization/trace.py) to generate the trace in Figure 1.

The pickled results outputted by [predict.py](predict.py) may also be fed to any of the scripts under [post_processing/](post_processing/) to generate the results referenced in III-B. These scripts will all output in the same format as [predict.py](predict.py) and may be fed to any of the prior mentioned evaluation scripts.

## Dependencies

All packages are available via pip or conda exception for pase which can be installed from [here](https://github.com/santi-pdp/pase). Reference [envrionment.yml](envrionment.yml) for specific versions.

- hmmlearn
- librosa
- matplotlib
- numpy
- pandas
- pase
- plotly
- pyannote-core
- pytorch
- scipy
- sklearn
- tqdm
- wandb

## Citation

TBD