/**
 * MCP Tool Marketplace — discover and catalog MCP servers.
 *
 * Provides a curated catalog of known MCP servers with metadata, install
 * commands, and category browsing.
 *
 * See: docs/05-the-tool-ecosystem/04-mcp-protocol.md
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ServerInfo {
  name: string;
  description: string;
  source: "npm" | "pypi" | "local";
  installCommand: string;
  runCommand: string[];
  toolCount: number;
  categories: string[];
  rating?: number;
  version: string;
}

// ---------------------------------------------------------------------------
// Curated catalog (Anthropic + community servers)
// ---------------------------------------------------------------------------

export const NPM_CATALOG: ServerInfo[] = [
  {
    name: "filesystem",
    description: "Read, write, search, and move files and directories.",
    source: "npm",
    installCommand: "npm install -g @modelcontextprotocol/server-filesystem",
    runCommand: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    toolCount: 9,
    categories: ["files", "utilities"],
    rating: 4.8,
    version: "latest",
  },
  {
    name: "github",
    description: "GitHub repository management: issues, PRs, branches, files.",
    source: "npm",
    installCommand: "npm install -g @modelcontextprotocol/server-github",
    runCommand: ["npx", "-y", "@modelcontextprotocol/server-github"],
    toolCount: 24,
    categories: ["development", "git"],
    rating: 4.7,
    version: "latest",
  },
  {
    name: "postgres",
    description: "PostgreSQL database querying with schema awareness.",
    source: "npm",
    installCommand: "npm install -g @modelcontextprotocol/server-postgres",
    runCommand: ["npx", "-y", "@modelcontextprotocol/server-postgres", "$DATABASE_URL"],
    toolCount: 3,
    categories: ["database"],
    rating: 4.6,
    version: "latest",
  },
  {
    name: "slack",
    description: "Send Slack messages and read channel history.",
    source: "npm",
    installCommand: "npm install -g @modelcontextprotocol/server-slack",
    runCommand: ["npx", "-y", "@modelcontextprotocol/server-slack"],
    toolCount: 5,
    categories: ["communication"],
    rating: 4.3,
    version: "latest",
  },
  {
    name: "brave-search",
    description: "Web and local search via the Brave Search API.",
    source: "npm",
    installCommand: "npm install -g @modelcontextprotocol/server-brave-search",
    runCommand: ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
    toolCount: 2,
    categories: ["search", "web"],
    rating: 4.5,
    version: "latest",
  },
  {
    name: "puppeteer",
    description: "Browser automation: navigate, screenshot, click, fill forms.",
    source: "npm",
    installCommand: "npm install -g @modelcontextprotocol/server-puppeteer",
    runCommand: ["npx", "-y", "@modelcontextprotocol/server-puppeteer"],
    toolCount: 6,
    categories: ["browser", "automation"],
    rating: 4.4,
    version: "latest",
  },
  {
    name: "google-maps",
    description: "Geocoding, directions, and places search via Google Maps.",
    source: "npm",
    installCommand: "npm install -g @modelcontextprotocol/server-google-maps",
    runCommand: ["npx", "-y", "@modelcontextprotocol/server-google-maps"],
    toolCount: 4,
    categories: ["maps", "utilities"],
    rating: 4.2,
    version: "latest",
  },
  {
    name: "memory",
    description: "Persistent key-value knowledge graph for long-term memory.",
    source: "npm",
    installCommand: "npm install -g @modelcontextprotocol/server-memory",
    runCommand: ["npx", "-y", "@modelcontextprotocol/server-memory"],
    toolCount: 5,
    categories: ["memory", "storage"],
    rating: 4.6,
    version: "latest",
  },
];

// ---------------------------------------------------------------------------
// MCPMarketplace
// ---------------------------------------------------------------------------

export class MCPMarketplace {
  private catalog: ServerInfo[];

  constructor(catalog: ServerInfo[] = NPM_CATALOG) {
    this.catalog = catalog;
  }

  /** Search by keyword in name or description. */
  search(query: string): ServerInfo[] {
    const q = query.toLowerCase();
    return this.catalog.filter(
      (s) =>
        s.name.includes(q) ||
        s.description.toLowerCase().includes(q) ||
        s.categories.some((c) => c.includes(q))
    );
  }

  /** List all categories present in the catalog. */
  listCategories(): string[] {
    const cats = new Set<string>();
    for (const s of this.catalog) s.categories.forEach((c) => cats.add(c));
    return [...cats].sort();
  }

  /** Get all servers in a category. */
  byCategory(category: string): ServerInfo[] {
    return this.catalog.filter((s) => s.categories.includes(category));
  }

  /** Get details for a named server. */
  getDetails(name: string): ServerInfo | undefined {
    return this.catalog.find((s) => s.name === name);
  }

  /** Sorted by rating (descending). */
  topRated(n = 5): ServerInfo[] {
    return [...this.catalog]
      .filter((s) => s.rating !== undefined)
      .sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0))
      .slice(0, n);
  }

  /** Print a formatted table to console. */
  printCatalog(): void {
    console.log("\n=== MCP Server Marketplace ===\n");
    console.log(`${"Name".padEnd(18)} ${"Category".padEnd(22)} ${"Tools".padEnd(7)} ${"Rating".padEnd(7)} Description`);
    console.log("─".repeat(80));
    for (const s of this.catalog) {
      const cat = s.categories.slice(0, 2).join(", ");
      const rating = s.rating !== undefined ? s.rating.toFixed(1) : "N/A";
      console.log(
        `${s.name.padEnd(18)} ${cat.padEnd(22)} ${s.toolCount.toString().padEnd(7)} ${rating.padEnd(7)} ${s.description.slice(0, 35)}`
      );
    }
    console.log("");
  }
}
