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

Examples of visualization:

