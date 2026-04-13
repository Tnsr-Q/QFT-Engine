.PHONY: all build test install docs clean

# Default: build the protoc-gen-doc binary.
all: build

# Build the protoc-gen-doc binary into ./bin/.
build:
	go build -o bin/protoc-gen-doc ./cmd/protoc-gen-doc

# Run all Go tests.
test:
	go test ./...

# Install protoc-gen-doc into $GOPATH/bin (or $GOBIN).
install:
	go install ./cmd/protoc-gen-doc

# Generate documentation from the bundled .proto files.
# Requires protoc to be installed and protoc-gen-doc on $PATH (run `make install` first).
#
# Usage:
#   make docs FORMAT=markdown OUTPUT=docs/api.md
#   make docs FORMAT=html     OUTPUT=docs/api.html
#   make docs FORMAT=json     OUTPUT=docs/api.json
FORMAT ?= markdown
OUTPUT ?= docs/api.md

docs: install
	@mkdir -p $(dir $(OUTPUT))
	protoc \
		--proto_path=proto \
		--doc_out=$(dir $(OUTPUT)) \
		--doc_opt=$(FORMAT),$(notdir $(OUTPUT)) \
		proto/tensorq/darwinian/v1/*.proto

clean:
	rm -rf bin/
