## Summary
Use `pytest.mark.parametrize` with pairs of an input and an expected context.
Use `pytest.raises` for an expected error and `contextlib.nullcontext` when the
operation should succeed. The same test body can then cover both outcomes.

## Generic example
```python
from contextlib import nullcontext as DoesNotRaise

import pytest


def validate_value(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


@pytest.mark.parametrize(
    argnames="value, error_context",
    argvalues=[
        ("example", DoesNotRaise()),
        ("", pytest.raises(expected_exception=ValueError, match="must not be blank")),
    ],
)
def test_validate_value(value, error_context):
    with error_context:
        validate_value(value=value)
```

Unexpected exceptions still fail the successful-input case. Matching the
message in the failing case helps reject an unrelated error of the same type.

---
Associations: [[python]], [[pytest]]
