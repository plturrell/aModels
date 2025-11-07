export default {
  root: "src",
  output: ".observablehq/dist",
  title: "Perplexity Dashboard",
  description: "Beautiful, interactive dashboard for exploring Perplexity processing results",
  pages: [
    {name: "Home", path: "/"},
    {name: "Processing", path: "/processing"},
    {name: "Results", path: "/results"},
    {name: "Analytics", path: "/analytics"},
    {name: "Knowledge Graph", path: "/graph"},
    {name: "Query", path: "/query"}
  ],
  theme: {
    primary: "#007AFF",
    background: "#ffffff",
    foreground: "#1d1d1f"
  }
};

