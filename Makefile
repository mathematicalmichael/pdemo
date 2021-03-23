build:
	docker build -t pdemo .

run:
	docker run --rm -d -p 8501:8501 pdemo
