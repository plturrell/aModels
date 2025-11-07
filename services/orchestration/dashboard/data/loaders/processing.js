/**
 * Data loader for processing status
 * Fetches processing request status from Perplexity API
 */

export default async function(requestId) {
  if (!requestId) {
    return null; // Return null instead of throwing for graceful handling
  }
  
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8000";
  const url = `${apiBase}/api/perplexity/status/${requestId}`;
  
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
      throw new Error(`Failed to fetch processing status: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error loading processing status:", error);
    // Return error object instead of throwing for graceful UI handling
    return {
      error: true,
      message: error.message || "Failed to load processing status",
      request_id: requestId
    };
  }
}

