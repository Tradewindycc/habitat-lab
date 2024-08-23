# Habitat-MAS

Habitat-MAS is a Python package for Multi-Agent Systems in Habitat virtual environments.

## Table of Contents
- [Habitat-MAS](#habitat-mas)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install habitat-mas](#install-habitat-mas)
  - [Usage](#usage)
    - [Download Data](#download-data)
    - [Data Structure](#data-structure)
    - [Run the demo](#run-the-demo)

## Installation

You should clone the our full fork of the habitat-lab project:

```sh
git clone https://github.com/SgtVincent/habitat-lab.git -b embodied_mas
```

### Prerequisites

Please make sure you have installed the [habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/v0.3.1), [habitat-lab](../README.md) and [habitat-baselines](../habitat-baselines/) and  following the normal habitat-lab installation guide.

**Note**: If you try to install the habitat suites on a Linux server without GUI, you may need to install headless version of the habitat-sim:

```sh
# Make sure nvidia-smi is available on your linux server
# $ sudo apt list --installed | grep nvidia-driver
# > nvidia-driver-xxx ...

conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```

### Install habitat-mas

To install the package, you can use the following command under habitat-lab root directory:

```sh
pip install -e habitat-mas
```

[Not available] Besides, you need to install the crab for multi-agent system.

## Usage

### Download Data
The dataset used in the demo is the same as [Habitat-3.0 Multi-Agent Training](../habitat-baselines/README.md#habitat-30-multi-agent-training). You can download the dataset by running the following command:

```sh
python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes habitat_humanoids hab_spot_arm hab3-episodes ycb hssd-hab hab3_bench_assets rearrange_task_assets
```

Besides, you should:
- download the robot configuration data from [here](https://drive.google.com/drive/folders/132Fhf0YGCEgMFUw93-b48eRiv4E9pj8h), and place it into the `data/robots` folder.
- download the perception, manipulation and mobility episodes data from [here](https://drive.google.com/drive/folders/1fnhzhRAW7Pzw48A4YsoijgdDPiKCArnK), and place them into the `data/datasets` folder.

The folder should look like this:
```
habitat-lab
├── data
│   ├── robots
│       ├── dji_drone
│           ├── meshes
│           ├── urdf
│       ├── hab_fetch
│       ├── hab_spot_arm
|           ├── meshesDae
│           ├── urdf
|               ├── hab_spot_onlyarm_dae.urdf
│               ...
│       ├── hab_stretch
│       ├── spot_data
│           ├── spot_walking_trajectory.csv
│       ├── json
│           ├── manipulation.json
│           ├── perception.json
│       ...
│   ├── datasets
│       ├── manipulation
│           ├── manipulation_eval_fetch.json.gz
│           ...
│       ├── perception
│           ├── perception_eval_drone.json.gz
│           ...
│       ├── mobility
│           ├── mobility_eval.json.gz
│           ...
...
```

### Dataset Generation

Here is a demo data generation command for dataset in `hssd` scene.

```sh
python habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --run \
--config data/config/replica_cad/all_object_receptacle_dataset.yaml \
--num-episodes 50 \
--out data/datasets/replica_cad/multi_objects_manipulation_eval.json.gz \
--type manipulation \
[Optional] --resume habitat-mas/habitat_mas/data/robot_resume/StretchRobot_default.json
```

`--config`: path of your dataset generation configuration data.

`--num-episodes`: episodes number you want to generate.

`--out`: desired path of your newly generated dataset.

`--type`: the purpose of your dataset, currently there are only three types: `preception`, `manipulation`, `hssd`.

`--resume`: path of desired robot resume as object position constraint, currently it is not used, you can freely ignore it.

Besides, for dataset generation, you should:

- download the dataset generation configuration data from [here](https://drive.google.com/drive/folders/1YVoCg2-tGkKWrdej4km6Abxsop0wS9XJ), and place them into the `data` folder.

### Dataset Structure

**Multi robot perception**

- Mainly deploy robots spot and drone for object perception.
- Objects are easy for drone to find, but hard for spot robot.

**Multi robot manipulation**

- Mainly deploy robots fetch and stretch for object rearrangement.
- Objects are easy for stretch robot to get, but hard for fetch robot.

### Run the demo

The demo is adapted from [Habitat-3.0 Social Rearrangement](../habitat-baselines/README.md#social-rearrangement). You can run the demo by running the following command:

```sh
# Under the habitat-lab root directory
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

Besides, to run our episodes, you should:

- download the high-level configuration data from [here](https://drive.google.com/drive/folders/1DR-WErfJLqmZuOCp1UUQ9T-scp8JdgPN), and place them into the `habitat-baselines/habitat_baselines/config` folder.
- download the simulator-level configuration data from [here](https://drive.google.com/drive/folders/1ovNky8ZzQVnVf_FyFaergRl3Qp94PWMz), and place them into the `habitat-lab/habitat/config/benchmark` folder.
- download the dataset configuration data from [here](https://drive.google.com/drive/folders/1bOM9aXEiifp-QL4w0GVj5qrGiU5ex0SI), and place them into the `habitat-lab/habitat/config/habitat/dataset`
