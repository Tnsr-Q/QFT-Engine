package plugin

import (
	"fmt"

	"google.golang.org/protobuf/types/descriptorpb"
)

// ProtoFile holds the extracted documentation for a single .proto file.
type ProtoFile struct {
	// Name is the file path as it appears in the proto descriptor.
	Name string `json:"name"`
	// Package is the protobuf package name.
	Package string `json:"package"`
	// Services defined in this file.
	Services []Service `json:"services,omitempty"`
	// Messages defined in this file (top-level).
	Messages []Message `json:"messages,omitempty"`
	// Enums defined in this file (top-level).
	Enums []Enum `json:"enums,omitempty"`
}

// Service describes a single protobuf service.
type Service struct {
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Methods     []Method `json:"methods,omitempty"`
}

// Method describes a single RPC method inside a service.
type Method struct {
	Name            string `json:"name"`
	Description     string `json:"description,omitempty"`
	InputType       string `json:"input_type"`
	OutputType      string `json:"output_type"`
	ClientStreaming bool   `json:"client_streaming,omitempty"`
	ServerStreaming bool   `json:"server_streaming,omitempty"`
}

// Message describes a protobuf message type.
type Message struct {
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Fields      []Field  `json:"fields,omitempty"`
	// Nested messages (message-in-message).
	Messages []Message `json:"messages,omitempty"`
	// Nested enums.
	Enums []Enum `json:"enums,omitempty"`
}

// Field describes a single field in a message.
type Field struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Type        string `json:"type"`
	Number      int32  `json:"number"`
	Label       string `json:"label"`
}

// Enum describes a protobuf enum type.
type Enum struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Values      []EnumValue `json:"values,omitempty"`
}

// EnumValue describes a single value inside an enum.
type EnumValue struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Number      int32  `json:"number"`
}

// ParseFile extracts documentation types from a FileDescriptorProto.
func ParseFile(fd *descriptorpb.FileDescriptorProto) ProtoFile {
	pf := ProtoFile{
		Name:    fd.GetName(),
		Package: fd.GetPackage(),
	}

	comments := buildCommentMap(fd)

	for i, sd := range fd.GetService() {
		svc := Service{
			Name:        sd.GetName(),
			Description: lookupComment(comments, []int32{6, int32(i)}),
		}
		for j, md := range sd.GetMethod() {
			svc.Methods = append(svc.Methods, Method{
				Name:            md.GetName(),
				Description:     lookupComment(comments, []int32{6, int32(i), 2, int32(j)}),
				InputType:       shortTypeName(md.GetInputType()),
				OutputType:      shortTypeName(md.GetOutputType()),
				ClientStreaming: md.GetClientStreaming(),
				ServerStreaming: md.GetServerStreaming(),
			})
		}
		pf.Services = append(pf.Services, svc)
	}

	for i, mt := range fd.GetMessageType() {
		pf.Messages = append(pf.Messages, parseMessage(mt, comments, []int32{4, int32(i)}))
	}

	for i, et := range fd.GetEnumType() {
		pf.Enums = append(pf.Enums, parseEnum(et, comments, []int32{5, int32(i)}))
	}

	return pf
}

func parseMessage(mt *descriptorpb.DescriptorProto, comments commentMap, path []int32) Message {
	msg := Message{
		Name:        mt.GetName(),
		Description: lookupComment(comments, path),
	}

	for i, f := range mt.GetField() {
		msg.Fields = append(msg.Fields, Field{
			Name:        f.GetName(),
			Description: lookupComment(comments, append(append([]int32{}, path...), 2, int32(i))),
			Type:        fieldTypeName(f),
			Number:      f.GetNumber(),
			Label:       labelName(f.GetLabel()),
		})
	}

	for i, nested := range mt.GetNestedType() {
		msg.Messages = append(msg.Messages, parseMessage(nested, comments, append(append([]int32{}, path...), 3, int32(i))))
	}

	for i, nested := range mt.GetEnumType() {
		msg.Enums = append(msg.Enums, parseEnum(nested, comments, append(append([]int32{}, path...), 4, int32(i))))
	}

	return msg
}

func parseEnum(et *descriptorpb.EnumDescriptorProto, comments commentMap, path []int32) Enum {
	e := Enum{
		Name:        et.GetName(),
		Description: lookupComment(comments, path),
	}

	for i, v := range et.GetValue() {
		e.Values = append(e.Values, EnumValue{
			Name:        v.GetName(),
			Description: lookupComment(comments, append(append([]int32{}, path...), 2, int32(i))),
			Number:      v.GetNumber(),
		})
	}

	return e
}

// commentMap maps a stringified source-code-info path to a comment string.
type commentMap map[string]string

func buildCommentMap(fd *descriptorpb.FileDescriptorProto) commentMap {
	m := make(commentMap)
	if fd.GetSourceCodeInfo() == nil {
		return m
	}
	for _, loc := range fd.GetSourceCodeInfo().GetLocation() {
		comment := cleanComment(loc.GetLeadingComments())
		if comment == "" {
			comment = cleanComment(loc.GetTrailingComments())
		}
		if comment == "" {
			continue
		}
		m[pathKey(loc.GetPath())] = comment
	}
	return m
}

func lookupComment(m commentMap, path []int32) string {
	return m[pathKey(path)]
}

func pathKey(path []int32) string {
	s := ""
	for i, p := range path {
		if i > 0 {
			s += "."
		}
		s += fmt.Sprintf("%d", p)
	}
	return s
}

func shortTypeName(fqn string) string {
	for i := len(fqn) - 1; i >= 0; i-- {
		if fqn[i] == '.' {
			return fqn[i+1:]
		}
	}
	return fqn
}

func fieldTypeName(f *descriptorpb.FieldDescriptorProto) string {
	switch f.GetType() {
	case descriptorpb.FieldDescriptorProto_TYPE_MESSAGE,
		descriptorpb.FieldDescriptorProto_TYPE_ENUM:
		return shortTypeName(f.GetTypeName())
	default:
		return scalarTypeName(f.GetType())
	}
}

func scalarTypeName(t descriptorpb.FieldDescriptorProto_Type) string {
	switch t {
	case descriptorpb.FieldDescriptorProto_TYPE_DOUBLE:
		return "double"
	case descriptorpb.FieldDescriptorProto_TYPE_FLOAT:
		return "float"
	case descriptorpb.FieldDescriptorProto_TYPE_INT64:
		return "int64"
	case descriptorpb.FieldDescriptorProto_TYPE_UINT64:
		return "uint64"
	case descriptorpb.FieldDescriptorProto_TYPE_INT32:
		return "int32"
	case descriptorpb.FieldDescriptorProto_TYPE_FIXED64:
		return "fixed64"
	case descriptorpb.FieldDescriptorProto_TYPE_FIXED32:
		return "fixed32"
	case descriptorpb.FieldDescriptorProto_TYPE_BOOL:
		return "bool"
	case descriptorpb.FieldDescriptorProto_TYPE_STRING:
		return "string"
	case descriptorpb.FieldDescriptorProto_TYPE_BYTES:
		return "bytes"
	case descriptorpb.FieldDescriptorProto_TYPE_UINT32:
		return "uint32"
	case descriptorpb.FieldDescriptorProto_TYPE_SFIXED32:
		return "sfixed32"
	case descriptorpb.FieldDescriptorProto_TYPE_SFIXED64:
		return "sfixed64"
	case descriptorpb.FieldDescriptorProto_TYPE_SINT32:
		return "sint32"
	case descriptorpb.FieldDescriptorProto_TYPE_SINT64:
		return "sint64"
	default:
		return "unknown"
	}
}

func labelName(l descriptorpb.FieldDescriptorProto_Label) string {
	switch l {
	case descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL:
		return "optional"
	case descriptorpb.FieldDescriptorProto_LABEL_REQUIRED:
		return "required"
	case descriptorpb.FieldDescriptorProto_LABEL_REPEATED:
		return "repeated"
	default:
		return ""
	}
}
