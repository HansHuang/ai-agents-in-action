// Entry point for code/go/02-structured-output.
//
// Use -mode to select which example to run:
//
//	-mode=template           Prompt template demo
//	-mode=few-shot          Zero-shot vs few-shot comparison
//	-mode=chain-of-thought  Reasoning with and without CoT
//	                        (docs/01-foundations/02-prompt-engineering.md)
//	-mode=structured        Structured output: schema-validated extraction
//	                        (docs/01-foundations/03-structured-output.md)
//
// Default is -mode=template.
package main

import (
	"flag"
	"fmt"
)

func main() {
	mode := flag.String("mode", "template", "Mode: template, few-shot, chain-of-thought, or structured")
	flag.Parse()

	switch *mode {
	case "template":
		runPromptTemplate()
	case "few-shot":
		RunFewShotComparison()
	case "chain-of-thought":
		RunChainOfThought()
	case "structured":
		runStructuredExtraction()
	default:
		fmt.Printf("Unknown mode %q. Use -mode=template, -mode=few-shot, -mode=chain-of-thought, or -mode=structured\n", *mode)
	}
}
