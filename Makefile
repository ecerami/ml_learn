PROJ_SLUG = ml
CLI_NAME = ml
PY_VERSION = 3.8
LINTER = flake8
FORMATTER = black -l 80 --experimental-string-processing

check: format lint test

freeze:
	pip freeze > requirements.txt

lint:
	$(LINTER) $(PROJ_SLUG)
	$(LINTER) tests

format:
	$(FORMATTER) $(PROJ_SLUG)
	$(FORMATTER) tests

qtest:
	py.test -s tests/

test:
	py.test -s --cov-report term --cov=$(PROJ_SLUG) tests/

coverage:
	py.test --cov-report html --cov=$(PROJ_SLUG) tests/

clean:
	rm -rf dist \
	rm -rf docs/build \
	rm -rf *.egg-info
	coverage erase

run_api:
	uvicorn luna.api.api:app --reload --host 0.0.0.0 --port 8000

run_api_prod:
	uvicorn luna.api.api:app --host 0.0.0.0 --port 8000
