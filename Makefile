GIT_VERSION = $(shell git describe --always --abbrev=7 --match=NeVeRmAtCh)
COMMIT_TIME = $(shell git log -1 --format=%cd)

PYTHON := $$(which python3)
PIP := $(PYTHON) -m pip

.PHONY: help
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

.PHONY: version
version: ## show up versions info of package
	@echo $(GIT_VERSION)
	@echo $(COMMIT_TIME)


.PHONY: install
install: ## install
	@$(PIP) install --no-cache-dir -r requirements.txt