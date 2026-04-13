package plugin

import (
	"fmt"
	"html"
	"strings"
)

// RenderHTML produces an HTML documentation page for the given proto files.
func RenderHTML(files []ProtoFile) (string, error) {
	var b strings.Builder

	b.WriteString(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Protocol Documentation</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem; color: #24292e; }
  h1 { border-bottom: 2px solid #e1e4e8; padding-bottom: 0.5rem; }
  h2 { border-bottom: 1px solid #e1e4e8; padding-bottom: 0.3rem; margin-top: 2rem; }
  h3, h4, h5 { margin-top: 1.5rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #e1e4e8; padding: 8px 12px; text-align: left; }
  th { background: #f6f8fa; font-weight: 600; }
  code { background: #f6f8fa; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
  .description { color: #586069; margin: 0.5rem 0; }
  .package { color: #586069; font-size: 0.9em; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; font-weight: 600; }
  .badge-stream { background: #dbedff; color: #0366d6; }
  hr { border: none; border-top: 1px solid #e1e4e8; margin: 2rem 0; }
</style>
</head>
<body>
<h1>Protocol Documentation</h1>
`)

	for _, f := range files {
		b.WriteString(fmt.Sprintf("<h2>%s</h2>\n", html.EscapeString(f.Name)))
		if f.Package != "" {
			b.WriteString(fmt.Sprintf("<p class=\"package\">Package: <code>%s</code></p>\n", html.EscapeString(f.Package)))
		}

		for _, svc := range f.Services {
			b.WriteString(fmt.Sprintf("<h3>Service: %s</h3>\n", html.EscapeString(svc.Name)))
			if svc.Description != "" {
				b.WriteString(fmt.Sprintf("<p class=\"description\">%s</p>\n", html.EscapeString(svc.Description)))
			}
			if len(svc.Methods) > 0 {
				b.WriteString("<table>\n<thead><tr><th>Method</th><th>Request</th><th>Response</th><th>Description</th></tr></thead>\n<tbody>\n")
				for _, m := range svc.Methods {
					reqType := html.EscapeString(m.InputType)
					respType := html.EscapeString(m.OutputType)
					if m.ClientStreaming {
						reqType = `<span class="badge badge-stream">stream</span> ` + reqType
					}
					if m.ServerStreaming {
						respType = `<span class="badge badge-stream">stream</span> ` + respType
					}
					desc := html.EscapeString(m.Description)
					b.WriteString(fmt.Sprintf("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n",
						html.EscapeString(m.Name), reqType, respType, desc))
				}
				b.WriteString("</tbody></table>\n")
			}
		}

		for _, msg := range f.Messages {
			writeMessageHTML(&b, msg, 3)
		}

		for _, e := range f.Enums {
			writeEnumHTML(&b, e, 3)
		}

		b.WriteString("<hr>\n")
	}

	b.WriteString("</body>\n</html>\n")
	return b.String(), nil
}

func writeMessageHTML(b *strings.Builder, msg Message, depth int) {
	tag := fmt.Sprintf("h%d", depth)
	if depth > 6 {
		tag = "h6"
	}
	b.WriteString(fmt.Sprintf("<%s>Message: %s</%s>\n", tag, html.EscapeString(msg.Name), tag))
	if msg.Description != "" {
		b.WriteString(fmt.Sprintf("<p class=\"description\">%s</p>\n", html.EscapeString(msg.Description)))
	}

	if len(msg.Fields) > 0 {
		b.WriteString("<table>\n<thead><tr><th>Field</th><th>Type</th><th>Label</th><th>Number</th><th>Description</th></tr></thead>\n<tbody>\n")
		for _, f := range msg.Fields {
			b.WriteString(fmt.Sprintf("<tr><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%s</td></tr>\n",
				html.EscapeString(f.Name),
				html.EscapeString(f.Type),
				html.EscapeString(f.Label),
				f.Number,
				html.EscapeString(f.Description)))
		}
		b.WriteString("</tbody></table>\n")
	}

	for _, nested := range msg.Messages {
		writeMessageHTML(b, nested, depth+1)
	}
	for _, nested := range msg.Enums {
		writeEnumHTML(b, nested, depth+1)
	}
}

func writeEnumHTML(b *strings.Builder, e Enum, depth int) {
	tag := fmt.Sprintf("h%d", depth)
	if depth > 6 {
		tag = "h6"
	}
	b.WriteString(fmt.Sprintf("<%s>Enum: %s</%s>\n", tag, html.EscapeString(e.Name), tag))
	if e.Description != "" {
		b.WriteString(fmt.Sprintf("<p class=\"description\">%s</p>\n", html.EscapeString(e.Description)))
	}

	if len(e.Values) > 0 {
		b.WriteString("<table>\n<thead><tr><th>Name</th><th>Number</th><th>Description</th></tr></thead>\n<tbody>\n")
		for _, v := range e.Values {
			b.WriteString(fmt.Sprintf("<tr><td>%s</td><td>%d</td><td>%s</td></tr>\n",
				html.EscapeString(v.Name), v.Number, html.EscapeString(v.Description)))
		}
		b.WriteString("</tbody></table>\n")
	}
}
