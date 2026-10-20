## Principle
Code structure should help maintainers understand how a component works.
Symmetric nesting or visually clean class boundaries do not by themselves make
responsibilities clear. If an abstraction hides the relationship between its
parts, reconsider the boundary rather than preserving symmetry for its own sake.

## Review questions
- Does the public interface match the mental model maintainers use?
- Can a reader identify which component owns each responsibility?
- Does an extra wrapper explain something, or merely make the structure look uniform?

---
Associations: [[Software Engineering]]
