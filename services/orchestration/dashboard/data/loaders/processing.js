/**
 * Data loader for processing status
 * Fetches processing request status from Perplexity API
 */

export default async function(requestId) {
  if (!requestId) {
    throw new Error("Request ID is required");
  }
  
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8080";
  const url = `${apiBase}/api/perplexity/status/${requestId}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch processing status: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error loading processing status:", error);
    throw error;
  }
}

