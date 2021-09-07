<p align="center">
    <br>
    <img src="https://repository-images.githubusercontent.com/401779782/c2f46be5-b74b-4620-ad64-57487be3b1ab" width="600"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/ElementAI/picard/actions/workflows/build.yml">
        <img alt="build" src="https://github.com/ElementAI/picard/actions/workflows/build.yml/badge.svg?branch=main&event=push">
    </a>
    <a href="https://github.com/ElementAI/picard/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/ElementAI/picard.svg?color=blue">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/tbd"><img alt="DOI" src="https://zenodo.org/badge/tbd.svg"></a>
</p>

This is the official implementation of the following paper:

[Torsten Scholak](https://twitter.com/tscholak), Nathan Schucher, Dzmitry Bahdanau. [PICARD - Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](). *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).*

## Overview

This code implements:

* The PICARD algorithm for constrained decoding from language models.
* A text-to-SQL semantic parser based on pre-trained sequence-to-sequence models and PICARD achieving state-of-the-art performance on both the [Spider](https://yale-lily.github.io/cosql) and the [CoSQL](https://yale-lily.github.io/cosql) datasets. 

## Methods

tbd.

## Quick Start

### Prerequisites

This repository uses git submodules. Clone it like this:
```sh
$ git clone git@github.com:ElementAI/picard.git
$ cd picard
$ git submodule update --init --recursive
```

### Docker

There are three docker images that can be used to run the code:

* **[tscholak/text-to-sql-dev](https://hub.docker.com/repository/docker/tscholak/text-to-sql-dev):** Base image with development dependencies. Use this for development. Pull it with `make pull-dev-image` from the docker hub. Rebuild the image with `make build-dev-image`. 
* **[tsscholak/text-to-sql-train](https://hub.docker.com/repository/docker/tscholak/text-to-sql-train):** Training image with development dependencies but without Picard dependencies. Use this for fine-tuning a model. Pull it with `make pull-train-image` from the docker hub. Rebuild the image with `make build-train-image`.
* **[tscholak/text-to-sql-eval](https://hub.docker.com/repository/docker/tscholak/text-to-sql-eval):** Training/evaluation image with all dependencies. Use this for evaluating a fine-tuned model with Picard. This image can also be used for training if you want to run evaluation during training with Picard. Pull it with `make pull-eval-image` from the docker hub. Rebuild the image with `make build-eval-image`.

All images are tagged with the current commit hash. The images are built with the buildx tool which is available in the latest docker-ce. Use `make init-buildkit` to initialize the buildx tool on your machine. You can then use `make build-dev-image`, `make build-train-image`, etc. to rebuild the images. Local changes to the code will not be reflected in the docker images unless they are committed to git.

### Training

The training script is located in `seq2seq/run_seq2seq.py`.
You can run it with:
```
$ make train
```
The model will be trained on the Spider dataset by default.
You can also train on CoSQL by running `make train-cosql`.

The training script will create the directory `train` in the current directory.
Training artifacts like checkpoints will be stored in this directory.

The default configuration is stored in `configs/train.json`.
The settings are optimized for a GPU with 40GB of memory.

### Evaluation

The evaluation script is located in `seq2seq/run_seq2seq.py`.
You can run it with:
```
$ make eval
```
By default, the evaluation will be run on the Spider evaluation set.
Evaluation on the CoSQL evaluation set can be run with `make eval-cosql`.

The evaluation script will create the directory `eval` in the current directory.
The evaluation results will be stored there.

The default configuration is stored in `configs/eval.json`.

## Pre-trained Models

#### Spider

<table>
  <tr>
    <th rowspan=2 valign=bottom>URL</th>
    <th colspan=2>Exact-set Match Accuracy</th>
    <th colspan=2>Execution Accuracy</th>
  </tr>
  <tr>
    <th>Dev</th>
    <th>Test</th>
    <th>Dev</th>
    <th>Test</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/tscholak/cxmefzzi">tscholak/cxmefzzi</a> <b>w PICARD</b></td>
    <td>75.5 %</td>
    <td>71.9 %</td>
    <td>79.3 %</td>
    <td>75.1 %</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/tscholak/cxmefzzi">tscholak/cxmefzzi</a> <b>w/o PICARD</b></td>
    <td>71.5 %</td>
    <td>68.0 %</td>
    <td>74.4 %</td>
    <td>70.1 %</td>
  </tr>
</table>

#### CoSQL Dialogue State Tracking

<table>
  <tr>
    <th rowspan=2 valign=bottom>URL</th>
    <th colspan=2>Question Match Accuracy</th>
    <th colspan=2>Interaction Match Accuracy</th>
  </tr>
  <tr>
    <th>Dev</th>
    <th>Test</th>
    <th>Dev</th>
    <th>Test</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/tscholak/2e826ioa">tscholak/2e826ioa</a> <b>w PICARD</b></td>
    <td>56.9 %</td>
    <td>??.? %</td>
    <td>24.2 %</td>
    <td>??.? %</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/tscholak/2e826ioa">tscholak/2e826ioa</a> <b>w/o PICARD</b></td>
    <td>53.8 %</td>
    <td>??.? %</td>
    <td>21.8 %</td>
    <td>??.? %</td>
  </tr>
</table>

## Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{Scholak2021:PICARD,
  author = {Torsten Scholak and Nathan Schucher and Dzmitry Bahdanau},
  title = {PICARD - Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models},
  booktitle = {EMNLP 2021},
  year = {2021},
}
```
