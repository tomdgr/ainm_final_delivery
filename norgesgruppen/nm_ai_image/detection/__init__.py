"""Detection module for NorgesGruppen shelf product detection.

Wraps ultralytics YOLOv8/RT-DETR for training and inference,
with embedding-based classification and WBF ensemble support.

Imports are lazy to avoid pulling in heavy dependencies (torch, lightning)
when only data/train modules are needed (e.g., on Azure ML).
"""
