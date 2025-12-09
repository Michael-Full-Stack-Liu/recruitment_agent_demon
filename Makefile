.PHONY: install test lint run build clean

install:
	pip install -r requirements.txt
	pip install pytest httpx flake8

test:
	python -m pytest tests/

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

run:
	python main.py

build:
	docker build -t recruitment-agent .

clean:
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	python -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
