import { useState, useEffect, useCallback, useRef } from 'react';
import { Search, RefreshCw, ArrowUpDown } from 'lucide-react';
import PostCard from './PostCard';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select } from './ui/select';
import { getPosts, getFeed } from '../api/client';
import { savePosts, getCachedPosts } from '../utils/db';

const SOURCES = [
  { value: '', label: 'All' },
  { value: 'reddit', label: 'Reddit' },
  { value: 'hackernews', label: 'HN' },
];

const SORT_OPTIONS = [
  { value: 'published_at', label: 'New' },
  { value: 'classifier_score', label: 'Top' },
];

const PAGE_SIZE = 20;
const FEED_LIMIT = 10;

export default function PostList({ isOffline = false, refreshKey = 0 }) {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hasMore, setHasMore] = useState(true);
  const [isFeedMode, setIsFeedMode] = useState(true);  // Track if using personalized feed

  // Filters
  const [source, setSource] = useState('');
  const [sort, setSort] = useState('published_at');
  const [order, setOrder] = useState('desc');
  const [search, setSearch] = useState('');
  const [unreadOnly, setUnreadOnly] = useState(false);

  const offsetRef = useRef(0);
  const observerRef = useRef(null);
  const loadingRef = useRef(null);

  const fetchPosts = useCallback(async (reset = false) => {
    if (reset) {
      offsetRef.current = 0;
      setHasMore(true);
    }

    setLoading(true);
    setError(null);

    try {
      let fetchedPosts;
      let hasNext = true;

      // Use personalized feed when no filters are applied
      const usePersonalizedFeed = !search && !source;
      setIsFeedMode(usePersonalizedFeed);

      if (isOffline) {
        fetchedPosts = await getCachedPosts({
          source: source || undefined,
          sort,
          order,
          limit: usePersonalizedFeed ? FEED_LIMIT : PAGE_SIZE,
          offset: offsetRef.current,
        });
        hasNext = !usePersonalizedFeed && fetchedPosts.length === PAGE_SIZE;
      } else {
        if (usePersonalizedFeed) {
          // Feed mode: get top 10 posts, no pagination
          const response = await getFeed({ limit: FEED_LIMIT, sort, order });
          fetchedPosts = response.items || [];
          hasNext = false;  // No infinite scroll for feed
        } else {
          // Filter mode: use regular posts endpoint with pagination
          const response = await getPosts({
            source: source || undefined,
            search: search || undefined,
            sort,
            order,
            limit: PAGE_SIZE,
            offset: offsetRef.current,
            unread_only: unreadOnly || undefined,
          });
          fetchedPosts = response.items || response.posts || [];
          hasNext = response.has_next ?? (fetchedPosts.length === PAGE_SIZE);
        }

        if (fetchedPosts.length > 0) {
          await savePosts(fetchedPosts);
        }
      }

      setHasMore(hasNext);

      if (reset || usePersonalizedFeed) {
        // Feed mode always replaces, filter mode appends on scroll
        setPosts(fetchedPosts);
      } else {
        setPosts((prev) => [...prev, ...fetchedPosts]);
      }

      offsetRef.current += fetchedPosts.length;
    } catch (err) {
      console.error('Failed to fetch posts:', err);

      // Try to fallback to cached posts
      if (!isOffline) {
        try {
          const cachedPosts = await getCachedPosts({
            source: source || undefined,
            sort,
            order,
            limit: PAGE_SIZE,
          });
          if (cachedPosts.length > 0) {
            setPosts(cachedPosts);
            // Don't show error if we have cached content
            setError(null);
            return;
          }
        } catch (cacheErr) {
          console.error('Cache fallback failed:', cacheErr);
        }
      }

      // Only show error if we couldn't load from cache
      setError(err.message || 'Failed to load posts');
    } finally {
      setLoading(false);
    }
  }, [isOffline, source, sort, order, search, unreadOnly]);

  // Initial load and filter changes
  useEffect(() => {
    fetchPosts(true);
  }, [source, sort, order, unreadOnly, refreshKey]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchPosts(true);
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  // Infinite scroll - only for filter mode, not feed mode
  useEffect(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    // Don't set up infinite scroll in feed mode
    if (isFeedMode) {
      return;
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loading) {
          fetchPosts(false);
        }
      },
      { threshold: 0.1 }
    );

    if (loadingRef.current) {
      observerRef.current.observe(loadingRef.current);
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [hasMore, loading, fetchPosts, isFeedMode]);

  const handlePostRead = useCallback((postId, isRead) => {
    setPosts((prev) =>
      prev.map((p) => (p.id === postId ? { ...p, is_read: isRead } : p))
    );
  }, []);

  return (
    <div className="space-y-3">
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="search"
          placeholder="Search..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* Compact Filter Row */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
        <Select
          value={source}
          onChange={(e) => setSource(e.target.value)}
          className="h-8 w-20 min-w-20 text-xs"
          aria-label="Source"
        >
          {SOURCES.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </Select>

        <Select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="h-8 w-16 min-w-16 text-xs"
          aria-label="Sort"
        >
          {SORT_OPTIONS.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </Select>

        <Button
          variant={order === 'desc' ? 'secondary' : 'ghost'}
          size="sm"
          className="h-8 w-8 p-0"
          onClick={() => setOrder(order === 'desc' ? 'asc' : 'desc')}
          aria-label="Order"
        >
          <ArrowUpDown className="h-3.5 w-3.5" />
        </Button>

        <label className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap text-xs">
          <input
            type="checkbox"
            checked={unreadOnly}
            onChange={(e) => setUnreadOnly(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-input"
          />
          Unread
        </label>

        <Button
          variant="ghost"
          size="sm"
          className="ml-auto h-8 w-8 p-0"
          onClick={() => fetchPosts(true)}
          disabled={loading}
          aria-label="Refresh"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {/* Posts */}
      <div className="space-y-2">
        {posts.map((post) => (
          <PostCard
            key={post.id}
            post={post}
            onRead={handlePostRead}
            isOffline={isOffline}
          />
        ))}
      </div>

      {/* Loading / End */}
      <div ref={loadingRef} className="flex justify-center py-6">
        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <RefreshCw className="h-4 w-4 animate-spin" />
            Loading...
          </div>
        )}
        {!loading && !hasMore && posts.length > 0 && (
          <span className="text-xs text-muted-foreground">End of feed</span>
        )}
        {!loading && posts.length === 0 && !error && (
          <span className="text-sm text-muted-foreground">No posts found</span>
        )}
      </div>
    </div>
  );
}
