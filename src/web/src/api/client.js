/**
 * API Client for News Aggregator
 * Handles all HTTP requests to the backend API
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

/**
 * Get stored user ID from localStorage
 */
function getUserId() {
  return localStorage.getItem('telegram_id') || '';
}

/**
 * Set user ID in localStorage
 */
export function setUserId(userId) {
  localStorage.setItem('telegram_id', userId);
}

/**
 * Base fetch wrapper with error handling and auth headers
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const userId = getUserId();

  const headers = {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true',
    ...options.headers,
  };

  if (userId) {
    headers['X-Telegram-ID'] = userId;
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(
        error.detail || `HTTP ${response.status}`,
        response.status,
        error
      );
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return null;
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    // Network error - might be offline
    throw new ApiError('Network error', 0, { offline: true });
  }
}

/**
 * Custom API Error class
 */
export class ApiError extends Error {
  constructor(message, status, details = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
    this.isOffline = details.offline || false;
  }
}

// =============================================================================
// Posts API
// =============================================================================

/**
 * Fetch posts with optional filters
 * @param {Object} params - Query parameters
 * @param {string} params.source - Filter by source (reddit, hackernews)
 * @param {string} params.search - Search query
 * @param {string} params.sort - Sort field (published_at, classifier_score)
 * @param {string} params.order - Sort order (asc, desc)
 * @param {number} params.limit - Number of posts to fetch
 * @param {number} params.offset - Pagination offset
 * @param {boolean} params.unread_only - Only show unread posts
 */
export async function getPosts(params = {}) {
  const searchParams = new URLSearchParams();
  
  if (params.source) searchParams.set('source', params.source);
  if (params.search) searchParams.set('search', params.search);
  if (params.sort) searchParams.set('sort', params.sort);
  if (params.order) searchParams.set('order', params.order);
  if (params.limit) searchParams.set('limit', params.limit.toString());
  if (params.offset) searchParams.set('offset', params.offset.toString());
  if (params.unread_only) searchParams.set('unread_only', 'true');

  const query = searchParams.toString();
  const endpoint = `/posts${query ? `?${query}` : ''}`;
  
  return apiFetch(endpoint);
}

/**
 * Fetch personalized feed based on user's topics and sources
 * Returns top N posts only (no pagination)
 * @param {Object} params - Query parameters
 * @param {number} params.limit - Max posts to return (default 10, max 20)
 * @param {number} params.min_score - Minimum classifier score (0-1)
 * @param {string} params.sort - Sort field (published_at, classifier_score)
 * @param {string} params.order - Sort order (asc, desc)
 */
export async function getFeed(params = {}) {
  const searchParams = new URLSearchParams();
  
  if (params.limit) searchParams.set('limit', params.limit.toString());
  if (params.min_score) searchParams.set('min_score', params.min_score.toString());
  if (params.sort) searchParams.set('sort', params.sort);
  if (params.order) searchParams.set('order', params.order);

  const query = searchParams.toString();
  const endpoint = `/posts/feed${query ? `?${query}` : ''}`;
  
  return apiFetch(endpoint);
}

/**
 * Get a single post by ID
 */
export async function getPost(postId) {
  return apiFetch(`/posts/${postId}`);
}

/**
 * Mark a post as read
 */
export async function markPostRead(postId) {
  return apiFetch(`/posts/${postId}/read`, {
    method: 'POST',
    body: JSON.stringify({ read: true }),
  });
}

/**
 * Mark a post as unread
 */
export async function markPostUnread(postId) {
  return apiFetch(`/posts/${postId}/read`, {
    method: 'POST',
    body: JSON.stringify({ read: false }),
  });
}

// =============================================================================
// Settings API
// =============================================================================

/**
 * Get user settings/preferences
 */
export async function getSettings() {
  return apiFetch('/settings');
}

/**
 * Update user settings/preferences
 * @param {Object} settings - Settings to update
 * @param {string[]} settings.topics - Interest topics
 * @param {string[]} settings.sources - Enabled sources
 * @param {number} settings.search_period_days - Search period in days
 */
export async function updateSettings(settings) {
  return apiFetch('/settings', {
    method: 'POST',
    body: JSON.stringify(settings),
  });
}

// =============================================================================
// Health API
// =============================================================================

/**
 * Check API health status
 */
export async function getHealth() {
  return apiFetch('/health');
}

// =============================================================================
// Export all functions
// =============================================================================

export default {
  getPosts,
  getFeed,
  getPost,
  markPostRead,
  markPostUnread,
  getSettings,
  updateSettings,
  getHealth,
  setUserId,
  ApiError,
};
