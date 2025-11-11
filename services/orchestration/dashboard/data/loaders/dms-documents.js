/**
 * Data loader for DMS documents list
 * Fetches documents from DMS API
 */

export default async function(options = {}) {
  const { limit = 50, offset = 0 } = options;
  
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8000";
  // DMS documents are available via gateway proxy to DMS service
  const dmsUrl = process.env.DMS_URL || "http://localhost:8096";
  const url = `${dmsUrl}/documents?limit=${limit}&offset=${offset}`;
  
  try {
    const response = await fetch(url, {
      headers: {
        "Accept": "application/json"
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch documents: ${response.statusText}`);
    }
    
    const documents = await response.json();
    
    return {
      documents: documents || [],
      total: documents?.length || 0,
      limit,
      offset
    };
  } catch (error) {
    console.error("Error loading DMS documents:", error);
    // Return error object instead of throwing for graceful UI handling
    return {
      error: true,
      message: error.message || "Failed to load documents",
      documents: [],
      total: 0
    };
  }
}

