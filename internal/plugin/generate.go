package plugin

import (
	"fmt"
	"strings"

	"google.golang.org/protobuf/types/pluginpb"
)

// Supported output formats.
const (
	FormatHTML     = "html"
	FormatMarkdown = "markdown"
	FormatJSON     = "json"
)

// Generate processes a CodeGeneratorRequest and returns a
// CodeGeneratorResponse containing the generated documentation files.
//
// The --doc_opt parameter (passed via the protoc plugin parameter field)
// controls the output format and file name:
//
//	--doc_opt=<format>,<output_file>
//
// Supported formats: html, markdown, json.
// If no parameter is provided the default is "markdown,docs.md".
func Generate(req *pluginpb.CodeGeneratorRequest) (*pluginpb.CodeGeneratorResponse, error) {
	format, outFile := parseParameter(req.GetParameter())

	// Build a set of files that the user explicitly asked protoc to compile
	// (as opposed to transitive imports).
	filesToGenerate := make(map[string]bool, len(req.GetFileToGenerate()))
	for _, f := range req.GetFileToGenerate() {
		filesToGenerate[f] = true
	}

	var protoFiles []ProtoFile
	for _, fd := range req.GetProtoFile() {
		if !filesToGenerate[fd.GetName()] {
			continue
		}
		protoFiles = append(protoFiles, ParseFile(fd))
	}

	var content string
	var err error
	switch format {
	case FormatHTML:
		content, err = RenderHTML(protoFiles)
	case FormatMarkdown:
		content, err = RenderMarkdown(protoFiles)
	case FormatJSON:
		content, err = RenderJSON(protoFiles)
	default:
		return nil, fmt.Errorf("unsupported output format %q (expected html, markdown, or json)", format)
	}
	if err != nil {
		return nil, err
	}

	resp := &pluginpb.CodeGeneratorResponse{
		File: []*pluginpb.CodeGeneratorResponse_File{
			{
				Name:    &outFile,
				Content: &content,
			},
		},
	}
	return resp, nil
}

// parseParameter extracts format and output file from the plugin parameter
// string.  Expected form: "<format>,<filename>" (e.g. "html,index.html").
func parseParameter(param string) (format, outFile string) {
	format = FormatMarkdown
	outFile = "docs.md"

	param = strings.TrimSpace(param)
	if param == "" {
		return
	}

	parts := strings.SplitN(param, ",", 2)
	if len(parts) >= 1 && parts[0] != "" {
		format = strings.ToLower(strings.TrimSpace(parts[0]))
	}
	if len(parts) >= 2 && parts[1] != "" {
		outFile = strings.TrimSpace(parts[1])
	} else {
		// Derive default file name from format.
		switch format {
		case FormatHTML:
			outFile = "docs.html"
		case FormatJSON:
			outFile = "docs.json"
		default:
			outFile = "docs.md"
		}
	}
	return
}
