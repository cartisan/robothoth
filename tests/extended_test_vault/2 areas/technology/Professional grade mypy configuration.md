Mypy's defaults are too lay, what you want is something like:
```
[tool.mypy]
disallow_untyped_defs = true
disallow_any_unimported = true
no_implicit_optional = true  
check_untyped_defs = true  
show_error_codes = true  
warn_unused_ignores = true  
warn_return_any = true  
```

### [disallow_untyped_defs = True](https://mypy.readthedocs.io/en/stable/config_file.html#confval-disallow_untyped_defs)
- enforce the usage of the type hints

### [disallow_any_unimported = True](https://mypy.readthedocs.io/en/stable/config_file.html#confval-disallow_any_unimported)
-  When something is imported from a dependency, it’s resolved to Any if mypy can’t resolve the import
- This gives us a warning instead

### [no_implicit_optional = True](https://mypy.readthedocs.io/en/stable/config_file.html#confval-no_implicit_optional)
- disallows: `def foo(arg: str = None) -> None`
- as the default here is not of type str we should have a `arg: str | None`

### [check_untyped_defs = True](https://mypy.readthedocs.io/en/stable/config_file.html#confval-check_untyped_defs)
- checks the body of untyped funtions

### [warn_return_any = True](https://mypy.readthedocs.io/en/stable/config_file.html#confval-warn_return_any)
- warns when a typed function returns a thing gtyped `Any` instead of its defined type

##
Associations: [[Software Engineering]]
Tags: #🌲 
Sources: https://careers.wolt.com/en/blog/tech/professional-grade-mypy-configuration
