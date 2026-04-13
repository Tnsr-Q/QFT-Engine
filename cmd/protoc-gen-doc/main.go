// protoc-gen-doc is a protoc plugin that generates documentation from .proto
// files in HTML, Markdown, or JSON format.
//
// Usage:
//
//	protoc --doc_out=./docs --doc_opt=markdown,api.md proto/*.proto
//
// The --doc_opt parameter controls the output:
//
//	--doc_opt=<format>,<output_file>
//
// Supported formats: html, markdown, json.
package main

import (
	"fmt"
	"os"

	"github.com/Tnsr-Q/QFT-Engine/internal/plugin"
)

func main() {
	if err := plugin.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "protoc-gen-doc: %v\n", err)
		os.Exit(1)
	}
}
