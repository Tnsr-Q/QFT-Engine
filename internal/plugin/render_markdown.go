package plugin

import (
	"fmt"
	"strings"
)

// RenderMarkdown produces Markdown documentation for the given proto files.
func RenderMarkdown(files []ProtoFile) (string, error) {
	var b strings.Builder

	b.WriteString("# Protocol Documentation\n\n")

	for _, f := range files {
		b.WriteString(fmt.Sprintf("## %s\n\n", f.Name))

		if f.Package != "" {
			b.WriteString(fmt.Sprintf("**Package:** `%s`\n\n", f.Package))
		}

		// Services
		for _, svc := range f.Services {
			b.WriteString(fmt.Sprintf("### Service: %s\n\n", svc.Name))
			if svc.Description != "" {
				b.WriteString(svc.Description + "\n\n")
			}

			if len(svc.Methods) > 0 {
				b.WriteString("| Method | Request | Response | Description |\n")
				b.WriteString("|--------|---------|----------|-------------|\n")
				for _, m := range svc.Methods {
					reqType := m.InputType
					respType := m.OutputType
					if m.ClientStreaming {
						reqType = "stream " + reqType
					}
					if m.ServerStreaming {
						respType = "stream " + respType
					}
					desc := strings.ReplaceAll(m.Description, "\n", " ")
					b.WriteString(fmt.Sprintf("| %s | %s | %s | %s |\n", m.Name, reqType, respType, desc))
				}
				b.WriteString("\n")
			}
		}

		// Messages
		for _, msg := range f.Messages {
			writeMessageMarkdown(&b, msg, 3)
		}

		// Enums
		for _, e := range f.Enums {
			writeEnumMarkdown(&b, e, 3)
		}

		b.WriteString("---\n\n")
	}

	return b.String(), nil
}

func writeMessageMarkdown(b *strings.Builder, msg Message, depth int) {
	heading := strings.Repeat("#", depth)
	b.WriteString(fmt.Sprintf("%s Message: %s\n\n", heading, msg.Name))
	if msg.Description != "" {
		b.WriteString(msg.Description + "\n\n")
	}

	if len(msg.Fields) > 0 {
		b.WriteString("| Field | Type | Label | Number | Description |\n")
		b.WriteString("|-------|------|-------|--------|-------------|\n")
		for _, f := range msg.Fields {
			desc := strings.ReplaceAll(f.Description, "\n", " ")
			b.WriteString(fmt.Sprintf("| %s | %s | %s | %d | %s |\n", f.Name, f.Type, f.Label, f.Number, desc))
		}
		b.WriteString("\n")
	}

	for _, nested := range msg.Messages {
		writeMessageMarkdown(b, nested, depth+1)
	}
	for _, nested := range msg.Enums {
		writeEnumMarkdown(b, nested, depth+1)
	}
}

func writeEnumMarkdown(b *strings.Builder, e Enum, depth int) {
	heading := strings.Repeat("#", depth)
	b.WriteString(fmt.Sprintf("%s Enum: %s\n\n", heading, e.Name))
	if e.Description != "" {
		b.WriteString(e.Description + "\n\n")
	}

	if len(e.Values) > 0 {
		b.WriteString("| Name | Number | Description |\n")
		b.WriteString("|------|--------|-------------|\n")
		for _, v := range e.Values {
			desc := strings.ReplaceAll(v.Description, "\n", " ")
			b.WriteString(fmt.Sprintf("| %s | %d | %s |\n", v.Name, v.Number, desc))
		}
		b.WriteString("\n")
	}
}
