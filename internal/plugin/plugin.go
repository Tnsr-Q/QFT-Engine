// Package plugin implements the protoc-gen-doc protoc plugin.
//
// It reads a CodeGeneratorRequest from stdin, extracts documentation from the
// proto file descriptors, and writes a CodeGeneratorResponse containing HTML,
// Markdown, or JSON documentation.
package plugin

import (
	"fmt"
	"io"
	"os"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/pluginpb"
)

// Run is the entry-point called from main.  It reads a CodeGeneratorRequest
// from stdin, generates documentation, and writes the response to stdout.
func Run() error {
	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		return fmt.Errorf("reading stdin: %w", err)
	}

	var req pluginpb.CodeGeneratorRequest
	if err := proto.Unmarshal(input, &req); err != nil {
		return fmt.Errorf("unmarshalling CodeGeneratorRequest: %w", err)
	}

	resp, err := Generate(&req)
	if err != nil {
		return fmt.Errorf("generating docs: %w", err)
	}

	out, err := proto.Marshal(resp)
	if err != nil {
		return fmt.Errorf("marshalling CodeGeneratorResponse: %w", err)
	}

	if _, err := os.Stdout.Write(out); err != nil {
		return fmt.Errorf("writing stdout: %w", err)
	}

	return nil
}
