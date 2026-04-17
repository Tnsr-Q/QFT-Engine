package plugin

import (
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/pluginpb"
)

// helper builds a minimal CodeGeneratorRequest with one file containing a
// service, message, and enum.
func testRequest(t *testing.T, param string) *pluginpb.CodeGeneratorRequest {
	t.Helper()

	fileName := "test.proto"
	pkg := "test.v1"
	svcName := "TestService"
	methodName := "GetWidget"
	inputType := ".test.v1.WidgetRequest"
	outputType := ".test.v1.WidgetResponse"
	msgName := "WidgetRequest"
	fieldName := "widget_id"
	fieldNum := int32(1)
	fieldType := descriptorpb.FieldDescriptorProto_TYPE_STRING
	fieldLabel := descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL
	enumName := "WidgetKind"
	enumVal0 := "WIDGET_KIND_UNSPECIFIED"
	enumVal1 := "WIDGET_KIND_ALPHA"
	enumNum0 := int32(0)
	enumNum1 := int32(1)
	serverStreaming := true

	fd := &descriptorpb.FileDescriptorProto{
		Name:    &fileName,
		Package: &pkg,
		Service: []*descriptorpb.ServiceDescriptorProto{
			{
				Name: &svcName,
				Method: []*descriptorpb.MethodDescriptorProto{
					{
						Name:            &methodName,
						InputType:       &inputType,
						OutputType:      &outputType,
						ServerStreaming:  &serverStreaming,
					},
				},
			},
		},
		MessageType: []*descriptorpb.DescriptorProto{
			{
				Name: &msgName,
				Field: []*descriptorpb.FieldDescriptorProto{
					{
						Name:   &fieldName,
						Number: &fieldNum,
						Type:   &fieldType,
						Label:  &fieldLabel,
					},
				},
			},
		},
		EnumType: []*descriptorpb.EnumDescriptorProto{
			{
				Name: &enumName,
				Value: []*descriptorpb.EnumValueDescriptorProto{
					{Name: &enumVal0, Number: &enumNum0},
					{Name: &enumVal1, Number: &enumNum1},
				},
			},
		},
	}

	return &pluginpb.CodeGeneratorRequest{
		FileToGenerate: []string{fileName},
		ProtoFile:      []*descriptorpb.FileDescriptorProto{fd},
		Parameter:      &param,
	}
}

func TestGenerateMarkdown(t *testing.T) {
	req := testRequest(t, "markdown,api.md")
	resp, err := Generate(req)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}

	if len(resp.GetFile()) != 1 {
		t.Fatalf("expected 1 file, got %d", len(resp.GetFile()))
	}
	f := resp.GetFile()[0]
	if f.GetName() != "api.md" {
		t.Errorf("expected file name api.md, got %s", f.GetName())
	}

	content := f.GetContent()
	for _, want := range []string{
		"# Protocol Documentation",
		"TestService",
		"GetWidget",
		"WidgetRequest",
		"widget_id",
		"WidgetKind",
		"WIDGET_KIND_ALPHA",
		"stream",
	} {
		if !strings.Contains(content, want) {
			t.Errorf("markdown output missing %q", want)
		}
	}
}

func TestGenerateHTML(t *testing.T) {
	req := testRequest(t, "html,docs.html")
	resp, err := Generate(req)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}

	f := resp.GetFile()[0]
	if f.GetName() != "docs.html" {
		t.Errorf("expected file name docs.html, got %s", f.GetName())
	}

	content := f.GetContent()
	for _, want := range []string{
		"<!DOCTYPE html>",
		"TestService",
		"GetWidget",
		"WidgetRequest",
		"widget_id",
		"WidgetKind",
		"badge-stream",
	} {
		if !strings.Contains(content, want) {
			t.Errorf("html output missing %q", want)
		}
	}
}

func TestGenerateJSON(t *testing.T) {
	req := testRequest(t, "json,docs.json")
	resp, err := Generate(req)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}

	f := resp.GetFile()[0]
	if f.GetName() != "docs.json" {
		t.Errorf("expected file name docs.json, got %s", f.GetName())
	}

	var files []ProtoFile
	if err := json.Unmarshal([]byte(f.GetContent()), &files); err != nil {
		t.Fatalf("JSON unmarshal error: %v", err)
	}

	if len(files) != 1 {
		t.Fatalf("expected 1 file, got %d", len(files))
	}
	pf := files[0]
	if pf.Name != "test.proto" {
		t.Errorf("expected file name test.proto, got %s", pf.Name)
	}
	if pf.Package != "test.v1" {
		t.Errorf("expected package test.v1, got %s", pf.Package)
	}
	if len(pf.Services) != 1 || pf.Services[0].Name != "TestService" {
		t.Error("unexpected services")
	}
	if len(pf.Messages) != 1 || pf.Messages[0].Name != "WidgetRequest" {
		t.Error("unexpected messages")
	}
	if len(pf.Enums) != 1 || pf.Enums[0].Name != "WidgetKind" {
		t.Error("unexpected enums")
	}
}

func TestDefaultParameter(t *testing.T) {
	param := ""
	req := testRequest(t, param)
	resp, err := Generate(req)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}
	f := resp.GetFile()[0]
	if f.GetName() != "docs.md" {
		t.Errorf("expected default file name docs.md, got %s", f.GetName())
	}
}

func TestUnsupportedFormat(t *testing.T) {
	param := "xml,docs.xml"
	req := testRequest(t, param)
	_, err := Generate(req)
	if err == nil {
		t.Fatal("expected error for unsupported format")
	}
	if !strings.Contains(err.Error(), "unsupported") {
		t.Errorf("expected 'unsupported' in error, got: %v", err)
	}
}

func TestRoundTrip(t *testing.T) {
	// Verify we can marshal/unmarshal the request through proto wire format.
	req := testRequest(t, "json,out.json")
	data, err := proto.Marshal(req)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}
	var req2 pluginpb.CodeGeneratorRequest
	if err := proto.Unmarshal(data, &req2); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}
	resp, err := Generate(&req2)
	if err != nil {
		t.Fatalf("Generate() after round-trip: %v", err)
	}
	if resp.GetFile()[0].GetName() != "out.json" {
		t.Errorf("unexpected file name after round-trip")
	}
}

func TestParseParameter(t *testing.T) {
	tests := []struct {
		input      string
		wantFormat string
		wantFile   string
	}{
		{"", "markdown", "docs.md"},
		{"html", "html", "docs.html"},
		{"json", "json", "docs.json"},
		{"markdown,api.md", "markdown", "api.md"},
		{"html,index.html", "html", "index.html"},
		{"json,service.json", "json", "service.json"},
		{"HTML,Docs.html", "html", "Docs.html"},
	}
	for _, tt := range tests {
		format, file := parseParameter(tt.input)
		if format != tt.wantFormat {
			t.Errorf("parseParameter(%q) format = %q, want %q", tt.input, format, tt.wantFormat)
		}
		if file != tt.wantFile {
			t.Errorf("parseParameter(%q) file = %q, want %q", tt.input, file, tt.wantFile)
		}
	}
}

func TestCleanComment(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{" This is a comment\n", "This is a comment"},
		{" Line 1\n Line 2\n", "Line 1\nLine 2"},
		{"", ""},
	}
	for _, tt := range tests {
		got := cleanComment(tt.input)
		if got != tt.want {
			t.Errorf("cleanComment(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestParseFileWithComments(t *testing.T) {
	// Build a FileDescriptorProto with source code info containing comments.
	fileName := "commented.proto"
	pkg := "commented.v1"
	svcName := "CommentedService"
	methodName := "DoStuff"
	inputType := ".commented.v1.Req"
	outputType := ".commented.v1.Resp"

	svcComment := " The CommentedService does stuff.\n"
	methodComment := " DoStuff performs the operation.\n"

	fd := &descriptorpb.FileDescriptorProto{
		Name:    &fileName,
		Package: &pkg,
		Service: []*descriptorpb.ServiceDescriptorProto{
			{
				Name: &svcName,
				Method: []*descriptorpb.MethodDescriptorProto{
					{
						Name:       &methodName,
						InputType:  &inputType,
						OutputType: &outputType,
					},
				},
			},
		},
		SourceCodeInfo: &descriptorpb.SourceCodeInfo{
			Location: []*descriptorpb.SourceCodeInfo_Location{
				{
					Path:            []int32{6, 0},
					LeadingComments: &svcComment,
				},
				{
					Path:            []int32{6, 0, 2, 0},
					LeadingComments: &methodComment,
				},
			},
		},
	}

	pf := ParseFile(fd)
	if len(pf.Services) != 1 {
		t.Fatalf("expected 1 service, got %d", len(pf.Services))
	}
	if pf.Services[0].Description != "The CommentedService does stuff." {
		t.Errorf("unexpected service description: %q", pf.Services[0].Description)
	}
	if len(pf.Services[0].Methods) != 1 {
		t.Fatalf("expected 1 method")
	}
	if pf.Services[0].Methods[0].Description != "DoStuff performs the operation." {
		t.Errorf("unexpected method description: %q", pf.Services[0].Methods[0].Description)
	}
}
