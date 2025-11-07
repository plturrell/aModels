/**
 * Data loader for relational intelligence data
 * Fetches aggregated intelligence from Orchestration API
 */

export default async function(requestId) {
  if (!requestId) {
    return null;
  }
  
  const apiBase = process.env.ORCHESTRATION_API_BASE || "http://localhost:8000";
  const url = `${apiBase}/api/relational/results/${requestId}/intelligence`;
  
  try {
    const response = await fetch(url, {
      headers: {
        "Accept": "application/json"
      }
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        return null;
      }
      throw new Error(`Failed to fetch relational intelligence: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error loading relational intelligence:", error);
    return {
      error: true,
      message: error.message || "Failed to load relational intelligence",
      request_id: requestId
    };
  }
}

