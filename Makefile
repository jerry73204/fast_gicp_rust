# Makefile for fast_gicp_rust project

.PHONY: build-no-cuda build-cuda test-no-cuda test-cuda clean help lint bench doc doc-rs generate-bindings

# Default target
help:
	@echo "Available targets:"
	@echo "  build-no-cuda    - Build without CUDA support"
	@echo "  build-cuda       - Build with CUDA support"
	@echo "  test-no-cuda     - Run tests without CUDA support"
	@echo "  test-cuda        - Run tests with CUDA support"
	@echo "  lint             - Run clippy linter"
	@echo "  bench            - Run benchmarks"
	@echo "  doc              - Generate documentation"
	@echo "  doc-rs           - Simulate docs.rs build (docs-only mode)"
	@echo "  generate-bindings - Generate pre-generated bindings for docs-only mode"
	@echo "  clean            - Clean build artifacts"
	@echo "  help             - Show this help message"

# Build without CUDA
build-no-cuda:
	cargo build --all-targets

# Build with CUDA
build-cuda:
	cargo build --all-targets --features cuda

# Test without CUDA
test-no-cuda:
	cargo nextest run --all-targets --no-fail-fast

# Test with CUDA (fallback to docs-only if CUDA not available)
test-cuda:
	cargo nextest run --all-targets --features cuda --no-fail-fast || cargo nextest run --lib --features docs-only --no-fail-fast

# Run linter
lint:
	cargo clippy --all-targets --all-features -- -D warnings

# Run benchmarks
bench:
	cargo bench

# Generate documentation
doc:
	cargo doc --no-deps --all-features --open

# Simulate docs.rs build (documentation-only mode without C++ dependencies)
doc-rs:
	@echo "Simulating docs.rs build environment..."
	cargo doc --no-default-features --features docs-only --no-deps
	@echo "Documentation generated in target/doc/"
	@echo "To view: open target/doc/fast_gicp/index.html"

# Generate pre-generated bindings for docs-only mode
generate-bindings:
	./scripts/generate-bindings.sh

# Clean build artifacts
clean:
	cargo clean