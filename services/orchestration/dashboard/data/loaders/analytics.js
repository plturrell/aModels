/**
 * Data loader for analytics data
 * Fetches analytics and trends from Perplexity API
 */

export default async function(options = {}) {
  const { limit = 100, offset = 0 } = options;
  
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8080";
  const url = `${apiBase}/api/perplexity/history?limit=${limit}&offset=${offset}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch analytics: ${response.statusText}`);
    }
    const data = await response.json();
    
    // Transform history data into analytics format
    return {
      requests: data.requests || [],
      total: data.total || 0,
      trends: calculateTrends(data.requests || [])
    };
  } catch (error) {
    console.error("Error loading analytics:", error);
    throw error;
  }
}

/**
 * Calculate trends from request history
 */
function calculateTrends(requests) {
  // Group by date
  const byDate = {};
  requests.forEach(req => {
    const date = new Date(req.created_at).toISOString().split('T')[0];
    if (!byDate[date]) {
      byDate[date] = { date, count: 0, success: 0, failed: 0 };
    }
    byDate[date].count++;
    if (req.status === 'completed') {
      byDate[date].success++;
    } else if (req.status === 'failed') {
      byDate[date].failed++;
    }
  });
  
  return Object.values(byDate).sort((a, b) => 
    new Date(a.date) - new Date(b.date)
  );
}

