# MGAvatar: Mesh-Bound Gaussians for Head Avatar Geometry and Appearance Modeling

<div align="center">

  <br>

[arxiv](#) / [video](media/video.mp4)

</div>

## Overview

We propose an end-to-end framework for high-fidelity head avatar reconstruction with mesh-bound Gaussians. In the geometry stage, FLAME parameters, vertex-bound Gaussians, and a pose-dependent deformation network are jointly optimized to refine identity-specific head geometry. In the appearance stage, face-bound Gaussians and Gaussian attribute offsets model fine-grained appearance, while a grid-based view-conditioned neural field improves color consistency under novel poses and expressions.

<div align="center">
  <img src="media/framework.png" width="100%">
</div>


## Setup

### [1. Installation](doc/installation.md)

### [2. Download](doc/download.md)

## Usage

### 1. Training

To train an MGAvatar model, simply run:

```shell
./run.sh
```

Please make sure that:

* The required dataset has been downloaded and preprocessed.
* The dataset paths are correctly configured.
* The required FLAME and tracking files are available.
* The environment has been installed successfully.

The training script will automatically perform the required optimization stages and save the trained model to the configured output directory.

### 2. Rendering

After training, use the following command to render the trained avatar:

```shell
./render.sh
```

The rendering script loads the trained MGAvatar model and generates the corresponding rendered results.

The output directory and rendering settings can be configured in `render.sh`.


## Acknowledgements

Our implementation is based on and inspired by several excellent open-source projects, including:

* [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
* [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* [NVDiffRec](https://github.com/NVlabs/nvdiffrec)
* [NVDiffRast](https://github.com/NVlabs/nvdiffrast)

We thank the authors for their valuable contributions to the community.

