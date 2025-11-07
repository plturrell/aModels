/**
 * Data loader for intelligence data
 * Fetches intelligence data (relationships, patterns, knowledge graph) from Perplexity API
 */

export default async function(requestId) {
  if (!requestId) {
    throw new Error("Request ID is required");
  }
  
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8000";
  const url = `${apiBase}/api/perplexity/results/${requestId}/intelligence`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch intelligence: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error loading intelligence:", error);
    throw error;
  }
}

