/**
 * Data loader for relational processing results
 * Fetches processed tables and results from Orchestration API
 */

export default async function(requestId) {
  if (!requestId) {
    return null;
  }
  
  const apiBase = process.env.ORCHESTRATION_API_BASE || "http://localhost:8000";
  const url = `${apiBase}/api/relational/results/${requestId}`;
  
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
      throw new Error(`Failed to fetch relational results: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error loading relational results:", error);
    return {
      error: true,
      message: error.message || "Failed to load relational results",
      request_id: requestId
    };
  }
}

