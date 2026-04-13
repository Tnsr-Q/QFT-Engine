package plugin

import (
	"fmt"
	"strings"
)

// cleanComment strips the leading space and trailing newline that protoc
// includes in source-code-info comments.
func cleanComment(s string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	var cleaned []string
	for _, l := range lines {
		cleaned = append(cleaned, strings.TrimPrefix(l, " "))
	}
	return strings.TrimSpace(strings.Join(cleaned, "\n"))
}

// pathKeyFmt is a helper for building comment map keys from variadic ints.
func pathKeyFmt(parts ...int32) string {
	s := ""
	for i, p := range parts {
		if i > 0 {
			s += "."
		}
		s += fmt.Sprintf("%d", p)
	}
	return s
}
