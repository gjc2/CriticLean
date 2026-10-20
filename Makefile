PYTHON ?= python3
.PHONY: install test smoke check build release github-dir clean

install:
	$(PYTHON) -m pip install '.[datasets,dev]'

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

smoke:
	PYTHONPATH=src $(PYTHON) scripts/smoke.py

check: test smoke

build:
	$(PYTHON) -m build

release:
	$(PYTHON) scripts/export_release.py --out releases/lean-critic-lab.zip

github-dir:
	$(PYTHON) scripts/export_release.py --github-dir releases/lean-critic-lab-github

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('build', 'dist', 'src/lean_critic_lab.egg-info')]"
