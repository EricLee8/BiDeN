# Codes and Data for BiDeN
The official code of our paper at EMNLP 2022: Back to the Future: Bidirectional Information Decoupling Network for Multi-turn Dialogue Modeling

## Environments and Dependencies
Our experiments are conducted on the following evironmental settings and dependencies. To ensure reproductivity, we strongly recommand that you run our code on the same settings.
- GPU: TITAN RTX 24G
- CUDA Version: 10.2

Note that the BiDeN contains 408M parameters, if your GPU memory is limited, you can lower the batch size (and decrease the learning rate proportionally to the decrease in batch size).

For dependencies, MuTual and Molweni dataset share the same dependencies:
- Pytorch Version: 1.6.0
- Python Version: 3.8.5
- Transformers Version: 4.6.0

DIALOGSUM and SAMSum are experimented under different versions:
- Pytorch Version: 1.11.0
- Python Version: 3.9.12
- Transformers Version: 4.18.0

## Usage
To install the dependencies for MuTual and Molweni, run:

`$ pip install -r requirements_molweni_mutual.txt`

For MuTual and MuTual_plus, you can download our best checkpoint at this [google drive](https://drive.google.com/drive/folders/16aHqO2-jH09AspBGZ7aIRCTIgdv41Dy1?usp=sharing).

To install the dependencies for DIALOGSUM and SAMSum, run:

`$ pip install -r requirements_summ.txt`

To run experiments on the MuTual and Molweni dataset with the default best hyper-parameter settings, run:

`$ cd [MuTual|Molweni]`

`$ python3 myTrain.py --model_file BiDeN (--dataset [mutual|mutual_plus])`

To run experiments on the DIALOGSUM and SAMSum dataset with the default best hyper-parameter settings, run:

`$ cd DialogueSumm`

`$ bash [run_dialogsum.sh|run_samsum.sh]`

Due to some stochastic factors(e.g., GPU and environment), it may need some slight tuning of the hyper-parameters using grid search to reproduce the results reported in our paper. Here are the suggested hyper-parameter settings:

### MuTual
- learning_rate: [3e-6, 5e-6, 6e-6, 8e-6, 1e-5]

### Molweni
- learning_rate: [3e-5, 5e-5, 7e-5, 8e-5, 1e-4]

### DIALOGSUM
- epochs: [8, 12, 15]
- learning_rate: [2e-5, 1e-5, 8e-6]

### SAMSum
- epochs: [3, 5, 10]
- learning_rate: [2e-5, 1e-5, 8e-6]

## Citation
If you find our paper or this repository useful, please cite us in your paper:
```
@inproceedings{li-etal-2022-back,
    title = "Back to the Future: Bidirectional Information Decoupling Network for Multi-turn Dialogue Modeling",
    author = "Li, Yiyang  and
      Zhao, Hai  and
      Zhang, Zhuosheng",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.177",
    pages = "2761--2774"
}
```
