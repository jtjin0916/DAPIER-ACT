# Model Card: DAPIER-ACT

## Intended use
Research on dual-arm tabletop manipulation with dual SO-101-class manipulators.

## Inputs
Task-frame head PointCloud, left wrist RGB, right wrist RGB, robot joint state.

## Output
A chunk of joint-space actions.

## Current evidence
Simulation evidence exists for controlled Pick -> Lift -> Return-to-Stow tasks.

## Not established
This package does not establish safe autonomous deployment, robust operation outside the trained workspace, collision-free behavior under distribution shift, real-world generalization, or reliable handling of arbitrary objects.

## Generalization claims
A model should only be described as generalized after held-out closed-loop evaluation across position, orientation and geometry. Larger training coverage alone is not proof of generalization.

## Safety
Always apply an independent hardware safety layer. Dry-run and low-speed staged validation are required before closed-loop actuation.
