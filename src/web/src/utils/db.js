/**
 * IndexedDB storage for offline posts
 * Uses the 'idb' library for a Promise-based API
 */

import { openDB } from 'idb';

const DB_NAME = 'news-aggregator';
const DB_VERSION = 1;

// Store names
const STORES = {
  POSTS: 'posts',
  SETTINGS: 'settings',
  PENDING_READS: 'pending-reads',
};

/**
 * Initialize the IndexedDB database
 */
async function initDB() {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db, oldVersion) {
      // Posts store
      if (!db.objectStoreNames.contains(STORES.POSTS)) {
        const postsStore = db.createObjectStore(STORES.POSTS, { keyPath: 'id' });
        postsStore.createIndex('source', 'source');
        postsStore.createIndex('published_at', 'published_at');
        postsStore.createIndex('classifier_score', 'classifier_score');
      }

      // Settings store (single document)
      if (!db.objectStoreNames.contains(STORES.SETTINGS)) {
        db.createObjectStore(STORES.SETTINGS, { keyPath: 'id' });
      }

      // Pending reads (to sync when back online)
      if (!db.objectStoreNames.contains(STORES.PENDING_READS)) {
        db.createObjectStore(STORES.PENDING_READS, { keyPath: 'postId' });
      }
    },
  });
}

// Singleton DB instance
let dbPromise = null;

function getDB() {
  if (!dbPromise) {
    dbPromise = initDB();
  }
  return dbPromise;
}

// =============================================================================
// Posts Operations
// =============================================================================

/**
 * Save posts to IndexedDB
 */
export async function savePosts(posts) {
  const db = await getDB();
  const tx = db.transaction(STORES.POSTS, 'readwrite');
  const store = tx.objectStore(STORES.POSTS);

  for (const post of posts) {
    await store.put({
      ...post,
      cached_at: new Date().toISOString(),
    });
  }

  await tx.done;
}

/**
 * Get all cached posts
 */
export async function getCachedPosts(options = {}) {
  const db = await getDB();
  const tx = db.transaction(STORES.POSTS, 'readonly');
  const store = tx.objectStore(STORES.POSTS);

  let posts = await store.getAll();

  // Filter by source
  if (options.source) {
    posts = posts.filter((p) => p.source === options.source);
  }

  // Sort
  const sortField = options.sort || 'published_at';
  const sortOrder = options.order || 'desc';
  posts.sort((a, b) => {
    const aVal = a[sortField] || 0;
    const bVal = b[sortField] || 0;
    return sortOrder === 'desc' ? (bVal > aVal ? 1 : -1) : (aVal > bVal ? 1 : -1);
  });

  // Pagination
  if (options.offset) {
    posts = posts.slice(options.offset);
  }
  if (options.limit) {
    posts = posts.slice(0, options.limit);
  }

  return posts;
}

/**
 * Get a single cached post by ID
 */
export async function getCachedPost(postId) {
  const db = await getDB();
  return db.get(STORES.POSTS, postId);
}

/**
 * Clear all cached posts
 */
export async function clearCachedPosts() {
  const db = await getDB();
  const tx = db.transaction(STORES.POSTS, 'readwrite');
  await tx.objectStore(STORES.POSTS).clear();
  await tx.done;
}

/**
 * Delete old cached posts (older than maxAge in milliseconds)
 */
export async function pruneOldPosts(maxAgeMs = 7 * 24 * 60 * 60 * 1000) {
  const db = await getDB();
  const tx = db.transaction(STORES.POSTS, 'readwrite');
  const store = tx.objectStore(STORES.POSTS);
  const posts = await store.getAll();
  
  const cutoff = new Date(Date.now() - maxAgeMs).toISOString();
  
  for (const post of posts) {
    if (post.cached_at < cutoff) {
      await store.delete(post.id);
    }
  }
  
  await tx.done;
}

// =============================================================================
// Settings Operations
// =============================================================================

const SETTINGS_KEY = 'user-settings';

/**
 * Save user settings to IndexedDB
 */
export async function saveSettings(settings) {
  const db = await getDB();
  await db.put(STORES.SETTINGS, { id: SETTINGS_KEY, ...settings });
}

/**
 * Get cached user settings
 */
export async function getCachedSettings() {
  const db = await getDB();
  const result = await db.get(STORES.SETTINGS, SETTINGS_KEY);
  if (result) {
    const { id, ...settings } = result;
    return settings;
  }
  return null;
}

// =============================================================================
// Pending Reads (Offline Sync Queue)
// =============================================================================

/**
 * Queue a post read action for later sync
 */
export async function queuePendingRead(postId, read = true) {
  const db = await getDB();
  await db.put(STORES.PENDING_READS, {
    postId,
    read,
    timestamp: new Date().toISOString(),
  });
}

/**
 * Get all pending read actions
 */
export async function getPendingReads() {
  const db = await getDB();
  return db.getAll(STORES.PENDING_READS);
}

/**
 * Remove a pending read action after successful sync
 */
export async function removePendingRead(postId) {
  const db = await getDB();
  await db.delete(STORES.PENDING_READS, postId);
}

/**
 * Clear all pending reads
 */
export async function clearPendingReads() {
  const db = await getDB();
  const tx = db.transaction(STORES.PENDING_READS, 'readwrite');
  await tx.objectStore(STORES.PENDING_READS).clear();
  await tx.done;
}

// =============================================================================
// Export
// =============================================================================

export default {
  savePosts,
  getCachedPosts,
  getCachedPost,
  clearCachedPosts,
  pruneOldPosts,
  saveSettings,
  getCachedSettings,
  queuePendingRead,
  getPendingReads,
  removePendingRead,
  clearPendingReads,
};
