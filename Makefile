PYTHON   := python
DIR      := models/morphpool
RUNS     := runs
STORAGES := storages

.PHONY: install clean reset

install:
	cd $(DIR);\
	$(PYTHON) setup.py install

reset: clean
	cd $(DIR);\
	rm -rf *.egg* dist build

clean:
	rm -rf $(RUNS) $(STORAGES)