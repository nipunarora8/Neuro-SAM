<div align="center">

# Neuro-SAM 
#### Foundation Model from Dendrite and Dendritic Spine Segmentation

</div>

Neuro-SAM enables you to:
- Trace individual dendrite in a 3D stack
- Segment traced dendrites using fine-tuned SAMv2
- Tubular View Analysis of the dendrites 
- Segment Dendritic Spines using our custom model

Neuro-SAM works across different imaging modalities including two-photon, confocal and STED microscopy.

### 🚀 Installation

Neuro-SAM requires **Python 3.10+** installed on your machine. It is recommended to use Conda/Miniconda for environment management. You can also use CUDA for GPU based accelerations. Our model are also optimised to use MPS on Apple Silicon (M series chips). 

To install Neuro-SAM: 

```bash
pip install neuro-sam
```

Downloading models and sample dataset

```bash
neuro-sam-download
```

### 📊 Usage

```bash
# base usage with benchmark dataset
neuro-sam

# using with your own dataset
neuro-sam --image-path /path/to/your/image.tif
```

### 🔬 Workflow

#### 1. **Configure Voxel Spacing**
Set accurate X, Y, Z voxel spacing in the "Path Tracing" tab for proper scaling:

#### 2. **Trace Dendritic Paths**
- Click waypoints along dendrite structures
- Algorithm automatically finds optimal brightest paths

#### 3. **Segment Dendrites**
- Load pre-trained SAMv2 dendrite model
- Segment individual path with SAMv2

#### 4. **Segment Spines**
- Segment Dendritic Spines with our fine tuned model

### 📬  Contact

- Nipun Arora - nipunarora8@yahoo.com
- Munna Singh - singhmunna0786@gmail.com


<div align="center">
<b style="color: black;">Made with ♥️ at <a href='https://anki.xyz' style="text-decoration: none; color: black;">Anki Lab</a> 🧠✨</b>
</div>
