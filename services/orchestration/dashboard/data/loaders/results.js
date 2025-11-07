/**
 * Data loader for processing results
 * Fetches processed documents and results from Perplexity API
 */

export default async function(requestId) {
  if (!requestId) {
    throw new Error("Request ID is required");
  }
  
  const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8080";
  const url = `${apiBase}/api/perplexity/results/${requestId}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch results: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error loading results:", error);
    throw error;
  }
}

