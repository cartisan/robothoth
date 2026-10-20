## Summary
 - The `query()` method allows you to extract `DataFrame` rows by specifying conditions through a query string, using comparison operators, string methods, logical combinations, and more. 
- It allows access to the local scope of the calling method in python.
- By default, `query()` returns a new `DataFrame`, and the original object remains unchanged.

## Examples
Using the following df:
```python
df = pd.read_csv('data/src/sample_pandas_normal.csv')
print(df)
#       name  age state  point
# 0    Alice   24    NY     64
# 1      Bob   42    CA     92
# 2  Charlie   18    CA     70
# 3     Dave   68    TX     70
# 4    Ellen   24    CA     88
# 5    Frank   30    NY     57
```
### Comparison
```python
print(df.query('age < 25'))  #equivalent to: df[df['age'] < 25]
#       name  age state  point
# 0    Alice   24    NY     64
# 2  Charlie   18    CA     70
# 4    Ellen   24    CA     88
```

Using variables from local scope:
```python
val = 25
print(df.query('age < @val'))
#       name  age state  point
# 0    Alice   24    NY     64
# 2  Charlie   18    CA     70
# 4    Ellen   24    CA     88
```
The advantage compared to example f-string is that it handles data types (dates, etc) and avoids escaping issues (special characters etc.)
### Set operations
```python
print(df.query('state in ["NY", "TX"]'))
#     name  age state  point
# 0  Alice   24    NY     64
# 3   Dave   68    TX     70
# 5  Frank   30    NY     57
```
### index operations
conditions for the `index` (row names) can be specified using the keyword `index`
```python
print(df.query('index % 2 == 0'))
#       name  age state  point
# 0    Alice   24    NY     64
# 2  Charlie   18    CA     70
# 4    Ellen   24    CA     88
```
### multiple conditions
* logical AND can be represented as either `&` or `and`
* logical or can be represented as either  `|` or `or`
* logical negations is represented as `not`
* parentheses for each condition are allowed but not necessary
```python
print(df.query('(age == 24 | point > 80) & state == "CA"'))
#     name  age state  point
# 1    Bob   42    CA     92
# 4  Ellen   24    CA     88
```

##
Associations: [[python]], [[pandas]]
Tags: #🌲 
Sources:
