# Driver Licensing Test

## Introduction of the project

### About

This project contains the source code and data for the paper "Behavioral Safety Assessment towards Large-scale Deployment of Autonomous Vehicles".

### Code structure

```
Driver-Licensing-Test
|- env: code used for analyze naturalistic driving data, generating test cases, run the test, and analyze the test results
|- traffic_signal: traffic signal control
|- vehicle: background vehicle control and vulnerable road user control
|- DLT.py: run the test of IDM model
|- get_risk_level_bounds.sh: generate risk level bounds of each scenario based on naturalistic driving dataset
|- NDD_analysis.sh: extract key parameters of each scenario
|- README.md
|- settings.py: settings of all functions
|- sim_analyze.py: analyze the test results of Autoware.Universe
|- tesla_data_postprocess.py: convert the raw data format of the test of Tesla
|- tesla_plot_results.py: plot test reuslts of Tesla and generate videos
|- utils.py: auxiliary functions
```

### Data

The datasets that are used in this project include [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html#download-link) and [rounD Dataset](https://levelxdata.com/round-dataset/). Please download them to anywhere you want.

The processed dataset, extracted data, test data, and analysis results can be found [here](https://zenodo.org/records/15150786). Please download it and copy the folders into the root of this repository.

## Usage

### Naturalistic driving dataset analysis (optional)

The first thing we need to do is analyze the naturalistic driving datasets (NDDs) and extract the the key parameters and initial conditions of all scenarios. The rounD Dataset is converted first according to the data processing method in [Learning Naturalistic Driving Environment with Statistical Realism](https://github.com/michigan-traffic-lab/Learning-Naturalistic-Driving-Environment?tab=readme-ov-file). Then run the following command:

```
bash NDD_analysis.sh
```

Then obtain the distribution of the key parameters and the initial conditions and get the risk level bounds:
```
bash get_risk_level_bounds.sh
```

The NDD processing lasts for hours. To save your time, the processed rounD dataset can be found [here](https://zenodo.org/records/15150786) in the folder `data/processed_rounD`. The extracted key parameters, initial conditions, and risk level bounds can be found [here](https://zenodo.org/records/15150786) in the folder `output/data_process`. Once the data is downloaded, this step can be skipped.

### Test conduction

The risk level bounds of each scenario are saved in `output/data_process/{scenario}/bounds.txt`. Then we need to copy the bounds into `env/route/Autoware.Universe/{scenario}/config.yaml`. To conduct the test of Autoware.Universe, please refer to the description in the [document](../README.md)

The detailed risk level bounds, generated test cases, and test results can be found [here](https://zenodo.org/records/15150786) in the folder `output/Autoware.Universe/risk_level`, `output/Autoware.Universe/case`, and `output/Autoware.Universe/test_data`, respectively.

### Test result analysis

The test results are saved at `output/Autoware.Universe/test_data`. Then run the following command to analyze the results:

```
python sim_analyze.py
```

The analysis results can be found [here](https://zenodo.org/records/15150786) in the folder `output/Autoware.Universe/evaluation`.

### Tesla testing data generation and process

Different test objectives share the same NDD analysis results in Driver Licensing Test, therefore, we don't need to do the NDD analysis again. To generate the risk levels and test cases for Tesla, please run the following code:
```
python DLT.py --test-name Tesla --scenario-folder left_turn_straight --case-num 10000
python DLT.py --test-name Tesla --scenario-folder vru_without_crosswalk --case-num 10000
```

The original test data format is different from that of the simulation, therefore, run the following command to post-process the test data:
```
python tesla_data_postprocess.py
```

Then use the following command to generate the figure and video of the test process of each case:
```
python tesla_plot_results.py
```

All results of the above steps, indluding generated risk levels, cases, original test data, processed test data, generated figures and videos can be found [here](https://zenodo.org/records/15150786) in the folder `output/Tesla/risk_level`, `output/Tesla/case`, `output/Tesla/original_test_data`, `output/Tesla/test_data`, `output/Tesla/figure`, respectively.
