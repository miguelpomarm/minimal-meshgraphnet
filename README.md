# Minimal MeshGraphNet

**A Minimal Graph Neural Network for Dynamical Systems with Spatial Distribution**

Bachelor's Thesis · University of Zaragoza (EINA) · Defended July 2026

- **Author**: Miguel Pomar Martínez
- **Supervisors**: Salvador Izquierdo Estallo, Alfonso Ortega Giménez
- **Degree**: Bachelor's in Telecommunications Engineering (specialization in Telecommunication Systems)
- **Full thesis (Spanish)**: [ZAGUÁN — TAZ-TFG-2026-3557](https://zaguan.unizar.es/record/TAZ-TFG-2026-3557)

---

## Abstract

This work proposes a simplified Graph Neural Network (GNN) architecture derived from DeepMind's MeshGraphNets [1], applied to fluid dynamics simulation on irregular meshes (CylinderFlow benchmark). 

The proposed model reinterprets the network's 128-dimensional latent space as a **discretised angular representation** around each node, distributing incoming messages among directional bins according to their geometric orientation. The optimal message distribution is computed via the Sinkhorn-Knopp algorithm, which replaces the learned aggregation MLP of the original architecture with a deterministic, physically motivated operation.

This reinterpretation enables a significant reduction in model size and computational cost while preserving predictive accuracy, and provides an explicit geometric meaning to each latent dimension.

## Key results

Ablation study on CylinderFlow, all models trained for 400k steps under identical conditions on a single NVIDIA T4 GPU:

| Model | Trainable params | RAM (MB) | FLOPs/frame | Training time (100k steps) |
|---|---:|---:|---:|---:|
| Reference `M = 15` | 2,332,930 | 8.90 | 31.36 G | 2h 23m |
| Reference `M = 5` | 845,570 | 3.23 | 11.11 G | 0h 57m |
| Intermediate (shared `MLP_edge`) | 484,098 | 1.85 | 7.73 G | 0h 43m |
| **Proposed** | **153,218** | **0.58** | **7.44 G** | **0h 50m** |

**Compared to the `M = 15` baseline, the proposed model reduces trainable parameters and memory footprint by 15× and inference FLOPs by 4×**, while maintaining rollout accuracy comparable to the `M = 5` reference.

Additionally, the proposed model's parameter count is **independent of the number of message-passing steps** `M`, thanks to weight sharing between processor blocks.

## Implementation

- **Language**: Python 3.7
- **Framework**: TensorFlow 1.15 with Sonnet
- **Infrastructure**: AWS SageMaker (`ml.g4dn.xlarge`, NVIDIA T4 GPU)
- **Dataset**: CylinderFlow (DeepMind), stored on Amazon S3

## Status

⚠️ **Code publication pending.** The source code will be released once approved by the thesis supervisors and the University of Zaragoza.

## Acknowledgements

This work was carried out with the guidance of the thesis supervisors. Development was assisted by AI tools (Claude, Cursor) for coding, debugging, and documentation.

## References

[1] T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P. W. Battaglia. *Learning Mesh-Based Simulation with Graph Networks*. ICLR 2021. [arXiv](https://arxiv.org/abs/2010.03409) — [DeepMind code](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)

## Contact
Miguel Pomar Martínez · [LinkedIn](https://linkedin.com/in/miguelpomarm) · [Email](mailto:miguelpomarm03@gmail.com)
