# Project 5 Methodology Design

## Selected Route

Design-science research plus experimental evaluation.

## Why Not Survey

A survey would measure opinions, not whether the system detects valid cable grasp points. For this topic, stronger evidence comes from segmentation metrics, centreline error, candidate validity, reachability and manipulation success if hardware exists.

## Evidence Levels

- Level 1: image perception metrics such as IoU, precision, recall, F1, centreline error and processing time.
- Level 2: grasp-candidate feasibility metrics such as endpoint distance, curvature, clearance and valid-candidate rate.
- Level 3: pose projection or simulation metrics if calibration/simulation exists.
- Level 4: physical grasp success only if robot trials exist.

## Chapter Four Gate

Chapter Four must not be generated until implementation outputs or experimental result files exist.

## Strengthened Protocol

The protocol is a standalone technical evaluation artifact, not a survey questionnaire. It now defines dataset registration, annotation rules, experiment matrix, metrics, failure-mode coding, result tables and Chapter Four handoff requirements.

## Figure Standard

Project 5 figures should use publication-style technical diagrams: high-resolution, source-adapted, readable at print size, captioned, and tied to the cited robotics/DLO literature or the project methodology. Avoid decorative AI-generated scenes or stock-style robotics images.
