## summary
Python-dotenv reads key-value pairs from a .env file and can set them as environment variables. It helps in the development of applications following the 12-factor principles. To configure the development environment, add a .env in the root directory of your project.
Syntax:
```
# Development settings
DOMAIN=example.org
ADMIN_EMAIL=admin@${DOMAIN}
ROOT_URL=${DOMAIN}/app
```
If you use variables in values, ensure they are surrounded with { and }, like ${DOMAIN}, as bare variables such as $DOMAIN are not expanded.

https://pypi.org/project/python-dotenv/

## commands
`dotenv run command` executes command## summary
Python-dotenv reads key-value pairs from a .env file and can set them as environment variables. It helps in the development of applications following the 12-factor principles. To configure the development environment, add a .env in the root directory of your project.
Syntax:
```
# Development settings
DOMAIN=example.org
ADMIN_EMAIL=admin@${DOMAIN}
ROOT_URL=${DOMAIN}/app
```
If you use variables in values, ensure they are surrounded with { and }, like ${DOMAIN}, as bare variables such as $DOMAIN are not expanded.

https://pypi.org/project/python-dotenv/

## commands
`dotenv run command` executes command
