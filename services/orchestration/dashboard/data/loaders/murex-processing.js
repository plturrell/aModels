/**
 * Data loader for Murex processing status
 * Fetches processing request status from Orchestration API
 */

export default async function(requestId) {
  if (!requestId) {
    return null;
  }
  
  const apiBase = process.env.ORCHESTRATION_API_BASE || "http://localhost:8000";
  const url = `${apiBase}/api/murex/status/${requestId}`;
  
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
      throw new Error(`Failed to fetch Murex processing status: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error loading Murex processing status:", error);
    return {
      error: true,
      message: error.message || "Failed to load Murex processing status",
      request_id: requestId
    };
  }
}

