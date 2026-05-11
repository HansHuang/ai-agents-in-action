// utils.go — Shared utility functions for the 03-agent-loop package.
package main

import "strings"

// repeatStr returns s repeated n times (like strings.Repeat).
func repeatStr(s string, n int) string {
	return strings.Repeat(s, n)
}
