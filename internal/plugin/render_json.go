package plugin

import (
	"encoding/json"
)

// RenderJSON serialises the parsed proto files to a JSON document.
func RenderJSON(files []ProtoFile) (string, error) {
	data, err := json.MarshalIndent(files, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data) + "\n", nil
}
