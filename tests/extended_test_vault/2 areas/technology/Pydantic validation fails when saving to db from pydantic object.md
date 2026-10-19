## Nullable values at a database boundary
A dictionary serialized from a Pydantic model can contain `None` values for
optional fields. A receiving database boundary may expect absent keys instead
of explicit nulls. When that is the contract, exclude null values during
serialization rather than copying a production model or schema into an example.

```python
payload = model.model_dump(exclude_none=True)
```

---
Associations: [[Pydantic]]
