# Project name
TAG = 0.1.0
IMAGE_NAME = cbt-assistant:$(TAG)
CONTAINER_NAME = cbt-assistant-dev

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) -f Dockerfile.dev .

# Run Docker container with GPU and mount current dir
run:
	docker run --gpus all -it -d --rm \
		--name $(CONTAINER_NAME) \
		-v $(shell pwd):/workspace \
		$(IMAGE_NAME)

# Rebuild image from scratch (no cache)
rebuild:
	docker build --no-cache -t $(IMAGE_NAME) .

# Remove image
clean:
	docker rmi -f $(IMAGE_NAME)

# Format code with black
format:
	black .
	isort .

# Run linters
lint:
	flake8 .

# Type checking
typecheck:
	mypy .

# Run all code checks
check: format lint typecheck
