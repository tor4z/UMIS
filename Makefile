PYTHON := python
DIR    := morphologicalpool
DEPS   := setup.py morphpool_cuda.cpp morphpool_cuda_kernel.cu

.PHONY: install clean

install: $(DEPS)
	cd $(DIR)
	$(PYTHON) setup.py install
	cd -

clean:
	cd $(DIR)
	rm -rf *.egg
	cd -