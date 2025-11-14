/**
 * Gitea Service API Client
 * 
 * Client for interacting with Gitea repository management endpoints
 */

const EXTRACT_SERVICE_URL = import.meta.env.VITE_EXTRACT_SERVICE_URL || 'http://localhost:8081';

export interface GiteaRepository {
  id: number;
  name: string;
  full_name: string;
  description: string;
  private: boolean;
  clone_url: string;
  ssh_url: string;
  html_url: string;
}

export interface CreateRepositoryRequest {
  owner?: string;
  name: string;
  description?: string;
  private?: boolean;
  auto_init?: boolean;
  readme?: string;
}

export interface GiteaBranch {
  name: string;
  commit: {
    id: string;
    message: string;
    author: {
      name: string;
      email: string;
      date: string;
    };
  };
}

export interface GiteaCommit {
  id: string;
  message: string;
  author: {
    name: string;
    email: string;
    date: string;
  };
  committer: {
    name: string;
    email: string;
    date: string;
  };
  url: string;
}

export interface GiteaFileInfo {
  name: string;
  path: string;
  sha: string;
  size: number;
  type: string; // 'file' or 'dir'
}

export interface FileContentResponse {
  path: string;
  content: string;
  ref: string;
}

export interface CloneRepositoryRequest {
  branch?: string;
  path?: string;
}

export interface CloneRepositoryResponse {
  clone_path: string;
  repository: GiteaRepository;
  branch: string;
}

export interface GiteaConfig {
  gitea_url?: string;
  gitea_token?: string;
}

/**
 * Build headers with Gitea config
 */
function buildHeaders(config?: GiteaConfig): HeadersInit {
  const headers: HeadersInit = {
    Accept: "application/json",
    "Content-Type": "application/json",
  };
  
  // Add Gitea config from parameter or environment
  const giteaURL = config?.gitea_url || import.meta.env.VITE_GITEA_URL;
  const giteaToken = config?.gitea_token || import.meta.env.VITE_GITEA_TOKEN;
  
  if (giteaURL) {
    headers["X-Gitea-URL"] = giteaURL;
  }
  if (giteaToken) {
    headers["X-Gitea-Token"] = giteaToken;
  }
  
  return headers;
}

/**
 * Build query string for non-Gitea config params
 */
function buildQueryString(params: Record<string, string | number | undefined> = {}): string {
  const queryParams = new URLSearchParams();
  
  // Add params (excluding Gitea config which goes in headers)
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && key !== 'gitea_url' && key !== 'gitea_token') {
      queryParams.append(key, String(value));
    }
  });
  
  const query = queryParams.toString();
  return query ? `?${query}` : '';
}

/**
 * Helper to fetch from extract service directly
 */
async function fetchFromExtractService<T>(
  endpoint: string, 
  init?: RequestInit,
  config?: GiteaConfig
): Promise<T> {
  const url = `${EXTRACT_SERVICE_URL}${endpoint}`;
  
  // Merge headers: Gitea config headers take precedence
  const giteaHeaders = buildHeaders(config);
  const mergedHeaders = {
    ...giteaHeaders,
    ...init?.headers,
  };
  
  try {
    const response = await fetch(url, {
      headers: mergedHeaders,
      ...init
    });

    if (!response.ok) {
      let errorMessage = `Request failed (${response.status})`;
      try {
        const errorText = await response.text();
        if (errorText) {
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorJson.message || errorText;
          } catch {
            errorMessage = errorText;
          }
        }
      } catch {
        errorMessage = `HTTP ${response.status} ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      const text = await response.text();
      if (!text.trim()) {
        return null as T;
      }
      return JSON.parse(text) as T;
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
        throw new Error(`Network error: Unable to reach ${url}. Check if the service is running and accessible.`);
      }
      throw error;
    }
    throw new Error(`Unexpected error: ${String(error)}`);
  }
}

/**
 * List repositories
 * @param owner - Optional owner name to filter repositories
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to array of repositories
 */
export async function listRepositories(
  owner?: string,
  config?: GiteaConfig
): Promise<GiteaRepository[]> {
  const query = buildQueryString({ owner });
  return fetchFromExtractService<GiteaRepository[]>(
    `/gitea/repositories${query}`,
    { method: 'GET' },
    config
  );
}

/**
 * Create a new repository
 * @param request - Repository creation request
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to created repository
 */
export async function createRepository(
  request: CreateRepositoryRequest,
  config?: GiteaConfig
): Promise<GiteaRepository> {
  return fetchFromExtractService<GiteaRepository>(
    `/gitea/repositories`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    },
    config
  );
}

/**
 * Get repository details
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to repository details
 */
export async function getRepository(
  owner: string,
  repo: string,
  config?: GiteaConfig
): Promise<GiteaRepository> {
  return fetchFromExtractService<GiteaRepository>(
    `/gitea/repositories/${owner}/${repo}`,
    { method: 'GET' },
    config
  );
}

/**
 * Delete a repository
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise that resolves when deletion is complete
 */
export async function deleteRepository(
  owner: string,
  repo: string,
  config?: GiteaConfig
): Promise<void> {
  return fetchFromExtractService<void>(
    `/gitea/repositories/${owner}/${repo}`,
    {
      method: 'DELETE',
    },
    config
  );
}

/**
 * List files in a repository directory
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param path - Optional directory path
 * @param ref - Optional branch/ref name (defaults to 'main')
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to array of file information
 */
export async function listFiles(
  owner: string,
  repo: string,
  path?: string,
  ref?: string,
  config?: GiteaConfig
): Promise<GiteaFileInfo[]> {
  const query = buildQueryString({
    path: path || '',
    ref: ref || 'main',
  });
  return fetchFromExtractService<GiteaFileInfo[]>(
    `/gitea/repositories/${owner}/${repo}/files${query}`,
    { method: 'GET' },
    config
  );
}

/**
 * Get file content
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param filePath - Path to the file
 * @param ref - Optional branch/ref name (defaults to 'main')
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to file content response
 */
export async function getFileContent(
  owner: string,
  repo: string,
  filePath: string,
  ref?: string,
  config?: GiteaConfig
): Promise<FileContentResponse> {
  const query = buildQueryString({
    ref: ref || 'main',
  });
  // Encode file path for URL
  const encodedPath = encodeURIComponent(filePath);
  return fetchFromExtractService<FileContentResponse>(
    `/gitea/repositories/${owner}/${repo}/files/${encodedPath}${query}`,
    { method: 'GET' },
    config
  );
}

/**
 * Create or update a file
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param path - File path
 * @param content - File content
 * @param message - Commit message
 * @param branch - Optional branch name (defaults to 'main')
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to operation result
 */
export async function createOrUpdateFile(
  owner: string,
  repo: string,
  path: string,
  content: string,
  message: string,
  branch?: string,
  config?: GiteaConfig
): Promise<{ path: string; message: string }> {
  return fetchFromExtractService<{ path: string; message: string }>(
    `/gitea/repositories/${owner}/${repo}/files`,
    {
      method: 'POST',
      body: JSON.stringify({
        path,
        content,
        message,
        branch: branch || 'main',
      }),
    },
    config
  );
}

/**
 * List branches
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to array of branches
 */
export async function listBranches(
  owner: string,
  repo: string,
  config?: GiteaConfig
): Promise<GiteaBranch[]> {
  return fetchFromExtractService<GiteaBranch[]>(
    `/gitea/repositories/${owner}/${repo}/branches`,
    { method: 'GET' },
    config
  );
}

/**
 * List commits
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param branch - Optional branch name (defaults to 'main')
 * @param limit - Optional limit on number of commits (defaults to 30, max 100)
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to array of commits
 */
export async function listCommits(
  owner: string,
  repo: string,
  branch?: string,
  limit?: number,
  config?: GiteaConfig
): Promise<GiteaCommit[]> {
  const query = buildQueryString({
    branch: branch || 'main',
    limit: limit?.toString(),
  });
  return fetchFromExtractService<GiteaCommit[]>(
    `/gitea/repositories/${owner}/${repo}/commits${query}`,
    { method: 'GET' },
    config
  );
}

/**
 * Clone repository for processing
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param request - Optional clone request (branch, path)
 * @param config - Optional Gitea configuration (URL and token)
 * @returns Promise resolving to clone response with clone path
 */
export async function cloneRepository(
  owner: string,
  repo: string,
  request?: CloneRepositoryRequest,
  config?: GiteaConfig
): Promise<CloneRepositoryResponse> {
  return fetchFromExtractService<CloneRepositoryResponse>(
    `/gitea/repositories/${owner}/${repo}/clone`,
    {
      method: 'POST',
      body: JSON.stringify(request || {}),
    },
    config
  );
}

