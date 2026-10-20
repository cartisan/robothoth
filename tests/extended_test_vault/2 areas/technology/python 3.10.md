## Structural Pattern Matching
- Like a case construct in e.g. Java, but relying on data structure analysis for complex matching:
```python
def func(person):  # person = (name, age, gender)
    match person:
        case (name, _, "male"):
            print(f"{name} is man.")
        case (name, _, "female"):
            print(f"{name} is woman.")
        case (name, age, gender):
            print(f"{name} is {age} old.")

func(("Example", 25, "male"))
# >> "Example is man."
```
- This is not limited to tuples, but can be used with Iterable or even classes, based on their attributes:
```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    gender: str

def func(person):  # person is instance of `Person` class
    match person:
        # This is not a constructor
        case Person(name, age, gender) if age < 18: # note the add. guard
            print(f"{name} is a child.")
        case Person(name=name, age=_, gender="male"):
            print(f"{name} is man.")
        case Person(name=name, age=_, gender="female"):
            print(f"{name} is woman.")
        case Person(name, age, gender):
            print(f"{name} is {age} years old.")
```

## Better Union Type Syntax
- can use `|` instead of having to import `Union`
```python
def func(value: int | float) -> int | float:
    return value
```

## Context Manager Syntax
-  Allows parenthesized context managers to span multiple lines



Associations: [[python]]
Tags: #🌲
