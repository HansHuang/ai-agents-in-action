// Entry point for code/go/02-structured-output.
//
// Use -mode to select which example to run:
//
//	-mode=template    Prompt engineering: templates, few-shot, chain-of-thought
//	                  (docs/01-foundations/02-prompt-engineering.md)
//	-mode=structured  Structured output: Pydantic-style extraction with retry
//	                  (docs/01-foundations/03-structured-output.md)
//
// Default is -mode=structured.
package main

import (
	"flag"
	"fmt"
)

func main() {
	mode := flag.String("mode", "structured", "Mode: template or structured")
	flag.Parse()

	switch *mode {
	case "template":
		runPromptTemplate()
	case "structured":
		runStructuredExtraction()
	default:
		fmt.Printf("Unknown mode %q. Use -mode=template or -mode=structured\n", *mode)
	}
}
