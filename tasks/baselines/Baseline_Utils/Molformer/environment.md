# Environment

The HELENA MolFormer comparison uses the shared root
[environment.yml](../../../../environment.yml). Follow the root
[setup instructions](../../../../README.md#setup-and-usage).

`tasks/baselines/Molformer_ensemble.py` uses the implementation in
`tasks/baselines/Baseline_Utils/GNN_modules.py` and the vocabulary in this
directory. The original upstream training, fine-tuning and notebook code is
retained for reference; its legacy Apex environment is not supported by the
shared HELENA setup.
