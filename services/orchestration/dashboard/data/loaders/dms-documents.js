/**
 * Data loader for documents list (migrated from DMS to Extract service)
 * Fetches documents from Extract service
 */

export default async function(options = {}) {
  const { limit = 50, offset = 0 } = options;
  
  // Use Extract service URL (replaces DMS)
  const extractUrl = process.env.EXTRACT_URL || process.env.DMS_URL || "http://localhost:8083";
  const url = `${extractUrl}/documents?limit=${limit}&offset=${offset}`;
  
  try {
    const response = await fetch(url, {
      headers: {
        "Accept": "application/json"
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch documents: ${response.statusText}`);
    }
    
    const data = await response.json();
    // Extract service returns { documents: [...], total, limit, offset }
    const documents = data.documents || data;
    
    return {
      documents: documents || [],
      total: data.total || documents?.length || 0,
      limit,
      offset
    };
  } catch (error) {
    console.error("Error loading documents:", error);
    // Return error object instead of throwing for graceful UI handling
    return {
      error: true,
      message: error.message || "Failed to load documents",
      documents: [],
      total: 0
    };
  }
}

