# Default target
.PHONY: all
all: lint test

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Main targets:"
	@echo "  all                 - Run lint and test (default)"
	@echo "  help                - Show this help message"
	@echo ""
	@echo "Stub management (both CUDA and non-CUDA together):"
	@echo "  generate-stubs      - Generate both CUDA and non-CUDA stubs"
	@echo "  verify-stubs        - Verify both stub variants (excludes/includes CUDA items)"
	@echo "  test-stubs          - Test both stub variants (docs-only with/without CUDA)"
	@echo ""
	@echo "Complete stub workflow:"
	@echo "  update-stubs        - Generate, verify, and test both stub variants"
	@echo ""
	@echo "Documentation:"
	@echo "  docs                - Generate docs using non-CUDA stub"
	@echo "  docs-cuda           - Generate docs using CUDA stub"
	@echo "  docs-compare        - Generate docs for both stub variants"
	@echo ""
	@echo "Development:"
	@echo "  test                - Run all tests (normal, CUDA, docs-only)"
	@echo "  lint                - Run all linting checks"
	@echo "  format              - Format code with rustfmt"
	@echo "  clean               - Clean build artifacts"
	@echo "  pre-commit          - Run all pre-commit checks"

# Generate both CUDA and non-CUDA stubs (requires C++ toolchain)
.PHONY: generate-stubs
generate-stubs:
	@echo "🔧 Generating both CUDA and non-CUDA stubs..."
	cargo build -p fast-gicp-sys --features bindgen
	@echo "✅ Generated both stub files:"
	@echo "  - fast-gicp-sys/src/generated/stub.rs (non-CUDA)"
	@echo "  - fast-gicp-sys/src/generated/stub_cuda.rs (CUDA)"

# Verify both stub variants are correct
.PHONY: verify-stubs
verify-stubs:
	@echo "🔍 Verifying both stub variants..."
	@echo "Checking non-CUDA stub excludes CUDA items..."
	@if grep -q "FastVGICPCuda\|NDTCuda" fast-gicp-sys/src/generated/stub.rs; then \
		echo "❌ ERROR: Non-CUDA stub contains CUDA items"; \
		exit 1; \
	else \
		echo "✅ Non-CUDA stub correctly excludes CUDA items"; \
	fi
	@echo "Checking CUDA stub includes CUDA items..."
	@if grep -q "FastVGICPCuda\|NDTCuda" fast-gicp-sys/src/generated/stub_cuda.rs; then \
		echo "✅ CUDA stub correctly includes CUDA items"; \
	else \
		echo "❌ ERROR: CUDA stub missing CUDA items"; \
		exit 1; \
	fi
	@echo "Generating expanded code for comparison..."
	cargo expand -p fast-gicp-sys --lib --features default > /tmp/expanded_default.rs
	cargo expand -p fast-gicp-sys --lib --features default,cuda > /tmp/expanded_cuda.rs
	@echo "✅ Verification complete - both stubs are correct"
	@echo "📄 Expanded code available at:"
	@echo "  - /tmp/expanded_default.rs (default features)"
	@echo "  - /tmp/expanded_cuda.rs (with CUDA)"

# Test both stub variants (compile only, since stubs panic when called)
.PHONY: test-stubs
test-stubs:
	@echo "🧪 Testing both stub variants (compile-only)..."
	@echo "Testing non-CUDA stub compilation (docs-only)..."
	cargo build --features docs-only --no-default-features --lib --tests
	cargo test --features docs-only --no-default-features --lib --no-run
	@echo "Testing CUDA stub compilation (docs-only + cuda)..."
	cargo build --features docs-only,cuda --no-default-features --lib --tests
	cargo test --features docs-only,cuda --no-default-features --lib --no-run
	@echo "✅ Both stub variants compile successfully"

# Complete stub workflow: generate, verify, and test both variants
.PHONY: update-stubs
update-stubs: generate-stubs verify-stubs test-stubs
	@echo ""
	@echo "🎉 Complete stub update workflow finished successfully!"
	@echo "Both CUDA and non-CUDA stubs have been:"
	@echo "  ✅ Generated from latest cxx::bridge definition"
	@echo "  ✅ Verified for correct CUDA item filtering"
	@echo "  ✅ Tested with docs-only builds"
	@echo ""
	@echo "Ready to commit the updated stub files."

# Run all tests
.PHONY: test
test:
	@echo "🧪 Running all tests..."
	cargo nextest run --all-targets --no-fail-fast
	cargo nextest run --features cuda --all-targets --no-fail-fast
	cargo nextest run --features docs-only --no-default-features --lib --tests --benches --no-fail-fast
	@echo "✅ All tests passed"

# Lint the code
.PHONY: lint
lint:
	cargo +nightly fmt --all -- --check
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features cuda -- -D warnings
	cargo clippy --lib --features docs-only --no-default-features -- -D warnings

# Format code
.PHONY: format
format:
	cargo +nightly fmt --all

# Clean build artifacts
.PHONY: clean
clean:
	cargo clean

# Generate documentation using non-CUDA stub (simulate docs.rs)
.PHONY: docs
docs:
	@echo "📚 Generating documentation using non-CUDA stub..."
	cargo doc --features docs-only --no-default-features --no-deps --open

# Generate documentation using CUDA stub
.PHONY: docs-cuda
docs-cuda:
	@echo "📚 Generating documentation using CUDA stub..."
	cargo doc --features docs-only,cuda --no-default-features --no-deps --open

# Compare documentation coverage between both stub variants
.PHONY: docs-compare
docs-compare:
	@echo "📚 Generating documentation for both stub variants..."
	@echo "Building non-CUDA documentation..."
	cargo doc --features docs-only --no-default-features --no-deps
	@echo "Building CUDA documentation..."
	cargo doc --features docs-only,cuda --no-default-features --no-deps
	@echo "✅ Documentation generated for comparison:"
	@echo "  - Non-CUDA: target/doc (docs-only)"
	@echo "  - CUDA: target/doc (docs-only + cuda)"

# Run all checks before committing
.PHONY: pre-commit
pre-commit: lint test verify-stubs
	@echo ""
	@echo "🚀 All pre-commit checks passed!"
	@echo "Ready to commit changes."
