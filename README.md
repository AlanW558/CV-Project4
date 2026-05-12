# CV-Project4

First, clone this repository and enter the project directory:

```bash
git clone https://github.com/AlanW558/CV-Project4.git
cd CV-Project4
```

## Part 1

### Environment Configuration

Part 1 requires COLMAP, 3DGS, VGGT, and Wavelet-GS.

For COLMAP and 3DGS, clone the official 3D Gaussian Splatting repository:

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
```

For VGGT, clone the official VGGT repository:

```bash
git clone https://github.com/facebookresearch/vggt
```

For Wavelet-GS, clone the Wavelet-GS repository:

```bash
git clone https://github.com/ALEX5874/Wavelet-GS.git
```

The detailed environment configuration methods should follow the instructions in the corresponding repositories:

* [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting.git)
* [VGGT](https://github.com/facebookresearch/vggt)
* [Wavelet-GS](https://github.com/ALEX5874/Wavelet-GS.git)

### Data Preparation

Create a `datasets` folder under the project directory:

```bash
mkdir datasets
```

The required datasets can be downloaded from the following links:

* [Google Drive](https://drive.google.com/drive/folders/1euG7pnbFowljVWoNLcbCmil81IVsIEfM)
* [Baidu Netdisk](https://pan.baidu.com/s/1Sa18zCeYiYA2gWAllo11dg?pwd=p3bm#list/path=%2F)

After downloading the datasets, place them into the `datasets` folder.

### Code Usage

After finishing the environment configuration and data preparation, run the corresponding scripts for Part 1.

Before running the code, make sure the external repositories and datasets are correctly prepared.

### Other Notes

Part 1 depends on COLMAP, 3D Gaussian Splatting, VGGT, and Wavelet-GS. Please refer to the official repositories for detailed installation instructions, dependency requirements, and additional usage notes.

## Part 2

### Environment Configuration

Configure the environment required for Part 2 according to the dependencies used in this part.

Make sure all required packages, datasets, and external resources are prepared before running the Part 2 code.

### Code Usage

When using VGGT, if the GPU memory is insufficient, you can uniformly sample the original datasets into subsampled datasets:

```bash
python scripts/generate_subsampled_datasets.py --frame 96
```

This script will read the datasets from `CV-Project4/datasets/` by default and save the subsampled datasets to:

```bash
CV-Project4/Part1/subsampled_datasets/
```

To run COLMAP initialization and 3DGS optimization, enter the Part 1 scripts folder:

```bash
cd CV-Project4/Part1/scripts
```

Then run:

```bash
python run_part1_colmap_3dgs.py --[dataset]
```

If you want to run COLMAP initialization and 3DGS optimization on the subsampled datasets, run:

```bash
python run_part1_colmap_3dgs.py --subsampled --frame 96 --[dataset]
```

Replace `--[dataset]` with one of the supported dataset options:

* `--waymo405841`
* `--dl3dv2`
* `--re10k1`

The outputs will be saved to the following folder by default:

```bash
CV-Project4/Part1/output/
```

For VGGT initialization and Wavelet-GS optimization, please refer to the corresponding official repositories for detailed usage instructions:

* [VGGT](https://github.com/facebookresearch/vggt)
* [Wavelet-GS](https://github.com/ALEX5874/Wavelet-GS.git)

After running all experiments, enter the Part 1 scripts folder:

```bash
cd CV-Project4/Part1/scripts
```

Then analyze the results using:

```bash
python part1_per_view_visualization.py
python part1_results_visualization.py
```

The visualizations will be saved to:

```bash
CV-Project4/Part1/figures/
```

### Other Notes

If Part 2 depends on the output from Part 1, make sure Part 1 has been completed successfully first.

## Part 3

### Environment Configuration

Configure the environment required for Part 3 according to the dependencies used in this part.

Make sure all necessary packages, model files, datasets, and configuration files are available before running the Part 3 code.

### Code Usage

Run the corresponding scripts for Part 3 from the project directory.

Before execution, check that the required inputs from previous parts are correctly generated and placed in the expected directories.

### Other Notes

If Part 3 uses results from Part 1 or Part 2, make sure the previous parts have been completed successfully before running this part.
