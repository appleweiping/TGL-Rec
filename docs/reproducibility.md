# Reproducibility

Every run should be reproducible from config and saved artifacts:

- control all seeds;
- save resolved configs;
- save environment and git metadata;
- write checkpoints for trainable smoke baselines;
- use one shared evaluator for all prediction rows;
- keep `outputs/` out of git;
- keep API keys out of configs and outputs.

Real API calls are disabled by default and must be explicitly approved.
