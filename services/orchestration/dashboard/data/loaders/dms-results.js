/**
 * Data loader for DMS processing results
 * Fetches processed documents and results from DMS API
 */

export default async function(requestId) {
  if (!requestId) {
    return null; // Return null instead of throwing for graceful handling
  }
  
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8000";
  const url = `${apiBase}/api/dms/results/${requestId}`;
  
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
      throw new Error(`Failed to fetch results: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error loading DMS results:", error);
    // Return error object instead of throwing for graceful UI handling
    return {
      error: true,
      message: error.message || "Failed to load results",
      request_id: requestId
    };
  }
}

