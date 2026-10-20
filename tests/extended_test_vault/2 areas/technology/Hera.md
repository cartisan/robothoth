## Summary
Hera creates and submits Argo workflows from Python.

## Scheduling
A workload on a spot instance may be interrupted. For a job that requires
on-demand capacity, request it with the public Kubernetes node-selector key
`karpenter.sh/capacity-type: on-demand`. The namespace, service connection, and
entry point below are placeholders for a local deployment.

```python
from hera.workflows import Workflow

with Workflow(
    generate_name="example-workflow-",
    entrypoint="example-step",
    namespace="<namespace>",
    node_selectors={"karpenter.sh/capacity-type": "on-demand"},
) as workflow:
    # Define the steps for this deployment here.
    pass
```

---
Associations: [[kubernetes]], [[z> MOC Infrastructure]]
