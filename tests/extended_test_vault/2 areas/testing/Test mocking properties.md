## Summary
Use `unittest.mock.PropertyMock` with `patch.object` to replace a property in a
test. This example defines a fictional configuration with no production imports.

```python
from unittest.mock import PropertyMock, patch


class ExampleConfig:
    @property
    def enabled(self):
        return False


def test_enabled_property():
    with patch.object(
        target=ExampleConfig,
        attribute="enabled",
        new_callable=PropertyMock,
        return_value=True,
    ):
        assert ExampleConfig().enabled is True
```

---
Associations: [[python]], [[unittest]]
