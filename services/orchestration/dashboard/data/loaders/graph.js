/**
 * Data loader for knowledge graph queries
 * Fetches knowledge graph data from Perplexity API
 */

export default async function loadGraph(requestId, query = null) {
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8080";
  
  try {
    // If query provided, execute it
    if (query) {
      const url = `${apiBase}/api/perplexity/graph/${requestId}/query`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, params: {} })
      });
      
      if (!response.ok) {
        throw new Error(`Graph query failed: ${response.statusText}`);
      }
      
      return await response.json();
    }
    
    // Otherwise, get relationships
    const url = `${apiBase}/api/perplexity/graph/${requestId}/relationships`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch relationships: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error loading graph data:", error);
    return { relationships: [], nodes: [], edges: [] };
  }
}

