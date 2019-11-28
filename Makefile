PYTHON   := python
DIR      := models/morphpool
RUNS     := runs
STORAGES := storages

.PHONY: install clean reset

install:
	cd $(DIR);\
	$(PYTHON) setup.py install

clean:
	cd $(DIR);\
	rm -rf *.egg* dist build

reset: clean
	rm -rf $(RUNS) $(STORAGES)