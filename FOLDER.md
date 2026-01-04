stateful-sequence-model/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── docs/
│   ├── theory.md
│   ├── derivations.md
│   ├── related_work.md
│   └── experiments.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── math/
│   │   ├── __init__.py
│   │   ├── state_space.py
│   │   ├── discretization.py
│   │   └── stability.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ssm_block.py
│   │   ├── gated_ssm.py
│   │   └── stack.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── optim.py
│   │   └── train.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py
│   │   └── long_context.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── memory_capacity.py
│   │   ├── scaling.py
│   │   └── benchmarks.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── seed.py
│
├── experiments/
│   ├── copy_task.yaml
│   ├── induction.yaml
│   └── recall.yaml
│
├── notebooks/
│   ├── stability_analysis.ipynb
│   ├── memory_decay.ipynb
│   └── attention_vs_ssm.ipynb
│
└── scripts/
    ├── run_copy_task.py
    ├── run_recall.py
    └── profile_scaling.py
