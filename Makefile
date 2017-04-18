MAINFILE=main

all: $(MAINFILE).py
	python3 $^



clean:
	rm -rf *~ $(PACKAGE)*.py *.pyc  __pycache*