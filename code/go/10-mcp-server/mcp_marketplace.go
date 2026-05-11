package main

import (
	"fmt"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// ServerInfo — describes one MCP server entry in the marketplace
// ---------------------------------------------------------------------------

// ServerInfo describes an MCP server available in the marketplace.
type ServerInfo struct {
	Name           string
	Description    string
	Source         string // "npm" | "pypi" | "local"
	InstallCommand string
	RunCommand     []string // argv to start the server
	ToolCount      int
	Categories     []string
	Rating         float64 // 0 = unrated
	Version        string
}

// ---------------------------------------------------------------------------
// Curated catalog
// ---------------------------------------------------------------------------

// npmCatalog is a small curated list of well-known MCP servers.
var npmCatalog = []ServerInfo{
	{
		Name:           "filesystem",
		Description:    "Read, write, search, and move files and directories.",
		Source:         "npm",
		InstallCommand: "npm install -g @modelcontextprotocol/server-filesystem",
		RunCommand:     []string{"npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
		ToolCount:      9,
		Categories:     []string{"files", "utilities"},
		Rating:         4.8,
		Version:        "latest",
	},
	{
		Name:           "github",
		Description:    "GitHub repository management: issues, PRs, branches, files.",
		Source:         "npm",
		InstallCommand: "npm install -g @modelcontextprotocol/server-github",
		RunCommand:     []string{"npx", "-y", "@modelcontextprotocol/server-github"},
		ToolCount:      20,
		Categories:     []string{"development", "vcs"},
		Rating:         4.7,
		Version:        "latest",
	},
	{
		Name:           "postgres",
		Description:    "Query and inspect PostgreSQL databases.",
		Source:         "npm",
		InstallCommand: "npm install -g @modelcontextprotocol/server-postgres",
		RunCommand:     []string{"npx", "-y", "@modelcontextprotocol/server-postgres"},
		ToolCount:      5,
		Categories:     []string{"databases"},
		Rating:         4.5,
		Version:        "latest",
	},
	{
		Name:           "brave-search",
		Description:    "Web and local search using the Brave Search API.",
		Source:         "npm",
		InstallCommand: "npm install -g @modelcontextprotocol/server-brave-search",
		RunCommand:     []string{"npx", "-y", "@modelcontextprotocol/server-brave-search"},
		ToolCount:      2,
		Categories:     []string{"search", "web"},
		Rating:         4.3,
		Version:        "latest",
	},
}

// pypiCatalog lists Python-based MCP servers available on PyPI.
var pypiCatalog = []ServerInfo{
	{
		Name:           "mcp-server-fetch",
		Description:    "Fetch web pages and convert to Markdown.",
		Source:         "pypi",
		InstallCommand: "pip install mcp-server-fetch",
		RunCommand:     []string{"python", "-m", "mcp_server_fetch"},
		ToolCount:      1,
		Categories:     []string{"web", "utilities"},
		Rating:         4.2,
		Version:        "latest",
	},
}

// ---------------------------------------------------------------------------
// MCPMarketplace
// ---------------------------------------------------------------------------

// MCPMarketplace catalogs, filters, and recommends MCP servers.
type MCPMarketplace struct {
	servers []ServerInfo
}

// NewMCPMarketplace creates a marketplace pre-loaded with the curated catalog.
func NewMCPMarketplace() *MCPMarketplace {
	all := make([]ServerInfo, 0, len(npmCatalog)+len(pypiCatalog))
	all = append(all, npmCatalog...)
	all = append(all, pypiCatalog...)
	return &MCPMarketplace{servers: all}
}

// Register adds a server to the marketplace.
func (m *MCPMarketplace) Register(s ServerInfo) {
	m.servers = append(m.servers, s)
}

// All returns all registered servers.
func (m *MCPMarketplace) All() []ServerInfo { return m.servers }

// FindByCategory returns servers whose category list contains the given tag.
func (m *MCPMarketplace) FindByCategory(category string) []ServerInfo {
	var results []ServerInfo
	for _, s := range m.servers {
		for _, c := range s.Categories {
			if strings.EqualFold(c, category) {
				results = append(results, s)
				break
			}
		}
	}
	return results
}

// Search returns servers whose name or description contains the query (case-insensitive).
func (m *MCPMarketplace) Search(query string) []ServerInfo {
	q := strings.ToLower(query)
	var results []ServerInfo
	for _, s := range m.servers {
		if strings.Contains(strings.ToLower(s.Name), q) ||
			strings.Contains(strings.ToLower(s.Description), q) {
			results = append(results, s)
		}
	}
	return results
}

// TopRated returns the top N servers sorted by rating descending.
func (m *MCPMarketplace) TopRated(n int) []ServerInfo {
	sorted := make([]ServerInfo, len(m.servers))
	copy(sorted, m.servers)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Rating > sorted[j].Rating
	})
	if n > len(sorted) {
		n = len(sorted)
	}
	return sorted[:n]
}

// InstallCommand returns the install command for a named server.
func (m *MCPMarketplace) InstallCommand(name string) (string, bool) {
	for _, s := range m.servers {
		if strings.EqualFold(s.Name, name) {
			return s.InstallCommand, true
		}
	}
	return "", false
}

// PrintCatalog prints the marketplace catalog to stdout.
func PrintCatalog(servers []ServerInfo) {
	if len(servers) == 0 {
		fmt.Println("(no servers found)")
		return
	}
	fmt.Printf("%-20s %-8s %-5s %-5s %s\n", "Name", "Source", "Tools", "Rating", "Description")
	fmt.Println(strings.Repeat("─", 80))
	for _, s := range servers {
		rating := "n/a"
		if s.Rating > 0 {
			rating = fmt.Sprintf("%.1f", s.Rating)
		}
		desc := s.Description
		if len(desc) > 45 {
			desc = desc[:42] + "..."
		}
		fmt.Printf("%-20s %-8s %-5d %-5s %s\n", s.Name, s.Source, s.ToolCount, rating, desc)
	}
}

// RunMCPMarketplaceDemo demonstrates the marketplace.
func RunMCPMarketplaceDemo() {
	mp := NewMCPMarketplace()

	fmt.Printf("MCP Marketplace — %d servers registered\n\n", len(mp.All()))

	fmt.Println("Top Rated:")
	PrintCatalog(mp.TopRated(3))

	fmt.Println("\nSearch 'search':")
	PrintCatalog(mp.Search("search"))

	fmt.Println("\nCategory 'files':")
	PrintCatalog(mp.FindByCategory("files"))

	if cmd, ok := mp.InstallCommand("filesystem"); ok {
		fmt.Printf("\nInstall filesystem: %s\n", cmd)
	}
}
