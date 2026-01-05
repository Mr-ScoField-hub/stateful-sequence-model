from src.math.state_space import StateSpaceModel
from src.math.stability import stability_report

ssm = StateSpaceModel(state_dim=16, input_dim=1)
report = stability_report(ssm.Lambda)

for k, v in report.items():
    print(k)
    print(v)
    print()
