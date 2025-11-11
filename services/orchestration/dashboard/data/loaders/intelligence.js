/**
 * Data loader for intelligence data
 * Fetches intelligence data (relationships, patterns, knowledge graph) from Perplexity API
 */

export default async function(requestId) {
  if (!requestId) {
    return null; // Return null instead of throwing for graceful handling
  }
  
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8000";
  const url = `${apiBase}/api/perplexity/results/${requestId}/intelligence`;
  
  try {
    const response = await fetch(url, {
      headers: {
        "Accept": "application/json"
      }
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        return null; // Request not found - return null for graceful handling
      }
      throw new Error(`Failed to fetch intelligence: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error loading intelligence:", error);
    // Return error object instead of throwing for graceful UI handling
    return {
      error: true,
      message: error.message || "Failed to load intelligence",
      request_id: requestId
    };
  }
}
