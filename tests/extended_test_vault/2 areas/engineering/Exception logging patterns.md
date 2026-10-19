## Big Tarp Pattern
- pessimistic approach: blanket-catch a whole range of possible exceptions (you might not know which), inform the log, and continue execution instead of terminating
```python
try:
    main_loop()
except Exception:
    logger.exception("Fatal error in main loop")
```

## Pinpoint Pattern
- optimistic approach: catch one specific exception, deal with it, and continue
```python
try:
    places = find_burrito_joints(criteria)
except BurritoCriteriaConflict as err:
    logger.warn(f"Cannot resolve conflicting burrito criteria: {err.message})
    places = list()
```

## Transformer Pattern
- catch an exception, log it and raise a different exception
- fitting at library boundaries, or when you want a package specific exception instead of a native one
- Important syntax:  `raise XYZError() from err`
```python
try:
    something()
except SomeError as err:
    logger.warn("...")
    raise DifferentError() from err
```

---
Associations: [[python]], [[Exception Handling]]
Tags: #🌲
