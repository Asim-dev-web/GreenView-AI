# GreenView-AI Training Metrics Summary

## Training Overview
- Model: U-Net++ with ResNet34 encoder (segmentation_models_pytorch)
- Phases:
  1. Warmup (5 epochs): Decoder-only training on LoveDA
  2. General (15 epochs): Full model training on LoveDA
  3. Adapt (15 epochs): Fine-tuning on mixed LoveDA + custom Connaught Place data
- Goal of phase 3: adapt to Indian urban scenes without catastrophic forgetting of LoveDA

## Why I Created My Own Dataset
Public datasets like LoveDA (mostly Chinese urban scenes) and others (DeepGlobe, Potsdam, etc.) have several limitations for Indian contexts:
- Domain shift: different building styles, vegetation patterns, lighting, and urban density cause poor generalization
- Lazy annotations: LoveDA masks often ignore individual trees, include barren land or water inside forest areas, or miss sparse greenery in cities
- No India-specific focus: almost no high-res urban Indian imagery, leading to false positives (green roofs, shadows) or missed detections

To fix this I manually annotated my own dataset using CVAT (Computer Vision Annotation Tool).  
I downloaded the high-resolution imagery from Google Earth Pro (centered on Connaught Place, New Delhi) and created 87 patches (512x512) with binary masks (0 = non-vegetation, 255 = vegetation).  
Annotation took many hours, but it was necessary to get meaningful adaptation.

## Phase 1 - Warmup (Decoder only on LoveDA)

| Epoch | Train Loss | Train IoU | LoveDA Val Loss | LoveDA Val IoU | Delhi Val Loss | Delhi Val IoU |
|-------|------------|-----------|-----------------|----------------|----------------|---------------|
| 1     | 1.0903     | 0.4668    | 1.0832          | 0.4668         | 0.9284         | 0.5806        |
| 2     | 1.0128     | 0.4970    | 1.0738          | 0.4667         | 0.8654         | 0.6022        |
| 3     | 0.9263     | 0.5272    | 1.0942          | 0.4601         | 0.7741         | 0.6259        |
| 4     | 0.9231     | 0.5396    | 1.1365          | 0.4489         | 0.9962         | 0.4583        |
| 5     | 0.9203     | 0.5410    | 1.1650          | 0.4428         | 0.9911         | 0.4530        |

## Phase 2 - General (Full model on LoveDA)

| Epoch | Train Loss | Train IoU | LoveDA Val IoU | Delhi Val IoU |
|-------|------------|-----------|----------------|---------------|
| 1     | 0.9138     | 0.5401    | 0.3081         | 0.2435        |
| 2     | 0.8628     | 0.5574    | 0.3992         | 0.4830        |
| 3     | 0.8416     | 0.5663    | nan            | 0.4082        |
| 4     | 0.8494     | 0.5550    | -              | 0.5230        |
| 5     | 0.8413     | 0.5744    | 0.2917         | 0.3751        |
| 6     | 0.8226     | 0.5706    | 0.4092         | 0.4617        |
| 7     | 0.8090     | 0.5742    | 0.4719         | 0.5227        |
| 8     | 0.7884     | 0.5860    | 0.1943         | 0.3264        |
| 9     | 0.7805     | 0.5840    | 0.3427         | 0.1946        |
| 10    | 0.7755     | 0.5882    | nan            | 0.2194        |
| 11    | 0.7809     | 0.5870    | 0.4295         | 0.4958        |
| 12    | 0.7727     | 0.5767    | 0.3707         | 0.5179        |
| 13    | 0.7590     | 0.5879    | 0.3446         | 0.2993        |
| 14    | 0.7484     | 0.5989    | 0.3801         | 0.3276        |
| 15    | 0.7414     | 0.5936    | 0.4211         | 0.4298        |

## Phase 3 - Adapt (Mixed LoveDA + Custom Delhi data)

| Epoch | Train Loss | Train IoU | LoveDA Val IoU | Delhi Val IoU |
|-------|------------|-----------|----------------|---------------|
| 1     | 0.8426     | 0.5716    | 0.3322         | 0.5902        |
| 2     | 0.7878     | 0.5921    | 0.3175         | 0.6003        |
| 3     | 0.7642     | 0.6036    | 0.3877         | 0.6007        |
| 4     | 0.7278     | 0.6201    | 0.4235         | 0.6152        |
| 5     | 0.7585     | 0.6001    | 0.3078         | 0.6230        |
| 6     | 0.7487     | 0.6024    | 0.3454         | 0.6236        |
| 7     | 0.7422     | 0.6128    | 0.3436         | 0.6249        |
| 8     | 0.6737     | 0.6538    | 0.3976         | 0.6304        |
| 9     | 0.6652     | 0.6581    | 0.3912         | 0.6325        |
| 10    | 0.7051     | 0.6354    | 0.3466         | 0.6352        |
| 11    | 0.7309     | 0.6168    | 0.3970         | 0.6387        |
| 12    | 0.6648     | 0.6477    | 0.3903         | 0.6413        |
| 13    | 0.6521     | 0.6522    | 0.3373         | 0.6373        |
| 14    | 0.6754     | 0.6374    | 0.3471         | 0.6404        |
| 15    | 0.6869     | 0.6360    | 0.4039         | 0.6412        |

## Key Observations
- Delhi validation IoU improved noticeably in Phase 3 (from roughly 0.43-0.52 range to peaking at 0.6412)
- LoveDA validation IoU fluctuates a lot around a mean of ~0.4 (goes up and down between 0.3 and 0.47) - this is expected when fine-tuning on a different domain
- Train IoU stayed relatively stable while Delhi performance increased - shows the adaptation is working
- LoveDA annotations are often lazy/inaccurate: when I tested my model on some LoveDA validation images, it actually detected vegetation better than the ground truth masks in several cases (e.g., picked up individual trees that were completely missed in the annotation, or correctly excluded barren land/water that was wrongly included inside forest areas)

Full raw log (including progress bars, interruptions, and warnings):  
[raw_training_log.txt](raw_training_log.txt)