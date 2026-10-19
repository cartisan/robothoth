## Self Type
- For type hints, previously, if we needed to refer to the current class itself, we’d have to explicitly define a type variable
- Now in 3.11, we can use a `Self` type, which refers to the encapsulating class itself.
```python
from typing import Any, Self

class Person:

    @classmethod
    def from_dict(cls, info: dict[str, Any]) -> Self:
        return Person(dict)
```

## TypeVarTuple
- Type annotation for arbitrarily sized tuples of generic types
- note that `*` is necessary when using `Ts`, like unpacking

```python
from typing import TypeVar, TypeVarTuple

T0 = TypeVar("T0")
Ts = TypeVarTuple("Ts")

def cycle(elements: tuple[T0, *Ts]) -> tuple[*Ts, T0]:
    first, *rest = elements
    return (*rest, first)
```

## Exception Group
- Several exceptions can be grouped into one and thrown together, like this:
```python
ExceptionGroup("twice", [TypeError("int"), ValueError(654)])
```
- In order to handle several exceptions, the keyword `except*` is introduced:
```python
try:
    raise ExceptionGroup("twice", [TypeError("int"), ValueError(654)])
except* ValueError as err:
    print(f"handling ValueError: {err.exceptions}")
except* TypeError as err:
    print(f"handling TypeError: {err.exceptions}")
```

## Exception Notes
- Additional notes can be added to an exception post-creation:
```python
err = ValueError(678)
err.add_note("Enriching Exceptions with Notes")
err.add_note("Python 3.11")
```

## Async Task Group
- Previously, asynchronous tasks needed to be manages in a list, and then manually `gather`-ed.
- Now, as task group context manager can be used for that:
```python
tasks = [asyncio.create_task(run_some_task(param)) for param in params]
await asyncio.gather(*tasks)

# now

async with asyncio.TaskGroup() as tg:
    for param in params:
        tg.create_task(run_some_task(param))
```

## Others
- faster code execution and start up
- more precise locations for error messages
- zero-cost exceptions


---
Associations: [[python]]
Tags: #🌲
