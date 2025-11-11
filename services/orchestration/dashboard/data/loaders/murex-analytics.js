/**
 * Data loader for Murex analytics data
 * Fetches history and calculates trends from Orchestration API
 */

export default async function() {
  const apiBase = process.env.ORCHESTRATION_API_BASE || "http://localhost:8000";
  const url = `${apiBase}/api/murex/history?limit=100`;
  
  try {
    const response = await fetch(url, {
      headers: {
        "Accept": "application/json"
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch Murex history: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    const requests = data.requests || [];
    
    // Calculate trends
    const now = new Date();
    const last24h = requests.filter(r => {
      if (!r.created_at) return false;
      const created = new Date(r.created_at);
      return (now - created) < 24 * 60 * 60 * 1000;
    });
    
    const last7d = requests.filter(r => {
      if (!r.created_at) return false;
      const created = new Date(r.created_at);
      return (now - created) < 7 * 24 * 60 * 60 * 1000;
    });
    
    const completed = requests.filter(r => r.status === "completed");
    const failed = requests.filter(r => r.status === "failed");
    
    return {
      requests,
      trends: {
        total: requests.length,
        last24h: last24h.length,
        last7d: last7d.length,
        completed: completed.length,
        failed: failed.length,
        success_rate: requests.length > 0 ? (completed.length / requests.length) * 100 : 0
      }
    };
  } catch (error) {
    console.error("Error loading Murex analytics:", error);
    return {
      error: true,
      message: error.message || "Failed to load Murex analytics",
      requests: [],
      trends: {}
    };
  }
}

