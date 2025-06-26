# Makefile for fast_gicp_rust project format

.PHONY: build-no-cuda build-cuda test-no-cuda test-cuda clean help lint format doc

# Default target
help:
	@echo "Available targets:"
	@echo "  build-no-cuda  - Build all targets without CUDA support"
	@echo "  build-cuda     - Build all targets with CUDA support"
	@echo "  test-no-cuda   - Run all tests without CUDA support"
	@echo "  test-cuda      - Run all tests with CUDA support"
	@echo "  lint           - Run clippy linter with strict warnings"
	@echo "  format         - Format code with rustfmt"
	@echo "  doc            - Generate documentation"
	@echo "  clean          - Clean build artifacts"
	@echo "  help           - Show this help message"

# Build without CUDA
build-no-cuda:
	cargo build --all-targets

# Build with CUDA
build-cuda:
	cargo build --all-targets --features cuda

# Test without CUDA
test-no-cuda:
	cargo nextest run --all-targets --no-fail-fast

# Test with CUDA
test-cuda:
	cargo nextest run --all-targets --features cuda --no-fail-fast

# Clean build artifacts
clean:
	cargo clean

lint:
	cargo clippy --all-targets --all-features -- -D warnings

format:
	cargo +nightly fmt

doc:
	cargo doc --no-deps --all-features
