## Motivation
Alembic versions database schemas and supports reviewed migrations instead of
dropping and rebuilding an existing database. It can compare SQLAlchemy schema
definitions with the database and generate a draft migration.

## Workflow
1. Rehearse the change in an isolated development environment.
2. Update the schema definitions.
3. Generate a migration with `alembic revision --autogenerate -m "example change"`.
4. Review the generated script, including the rollback path where reversible.
5. Run `alembic upgrade head`; use `alembic downgrade -1` for a supported rollback.
6. Configure the deployment runner with an authorized connection reference,
   represented here by `<database-connection>`, rather than embedding credentials.

The `alembic_version` table records the current schema revision. A migration can
use a database connection to perform a reviewed data transformation when needed.

---
Associations: [[python]], [[database]]
