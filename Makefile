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

submit_kaggle_twitter:
	kaggle competitions submit -c nlp-getting-started -f out/twitter_predictions.csv -m "Latest"