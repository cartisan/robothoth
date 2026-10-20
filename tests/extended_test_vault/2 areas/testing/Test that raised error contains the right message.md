## Summary
Checking only the right type of error is often insufficient: different failures
can raise the same exception class. Use `match` to require the expected message.

## Generic example
```python
import pytest


def validate_value(value: str) -> None:
    if not value.strip():
        raise ValueError("value must not be blank")


def test_blank_value():
    with pytest.raises(expected_exception=ValueError, match=r"^value must not be blank$"):
        validate_value(value="")
```

This rejects an unrelated `ValueError` with a different message.

---
Associations: [[python]], [[pytest]]
