import { useState, useCallback } from 'react';
import { ExternalLink, Clock, TrendingUp, Check, Circle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardFooter } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { markPostRead, markPostUnread } from '../api/client';
import { queuePendingRead } from '../utils/db';

function formatRelativeTime(dateString) {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'now';
  if (diffMins < 60) return `${diffMins}m`;
  if (diffHours < 24) return `${diffHours}h`;
  if (diffDays < 7) return `${diffDays}d`;
  return date.toLocaleDateString();
}

function getSourceInfo(sourceName) {
  if (!sourceName) return { label: 'Unknown', color: 'secondary' };
  
  if (sourceName === 'hackernews') {
    return { label: 'HN', color: 'default' };
  }
  
  if (sourceName.startsWith('reddit/r/')) {
    const sub = sourceName.replace('reddit/r/', 'r/');
    return { label: sub, color: 'secondary' };
  }
  
  if (sourceName === 'reddit') {
    return { label: 'Reddit', color: 'secondary' };
  }
  
  return { label: sourceName, color: 'secondary' };
}

export default function PostCard({ post, onRead, isOffline = false }) {
  const [isUpdating, setIsUpdating] = useState(false);

  const handleToggleRead = useCallback(async () => {
    if (isUpdating) return;

    const newReadState = !post.is_read;
    setIsUpdating(true);
    onRead?.(post.id, newReadState);

    try {
      if (isOffline) {
        await queuePendingRead(post.id, newReadState);
      } else {
        if (newReadState) {
          await markPostRead(post.id);
        } else {
          await markPostUnread(post.id);
        }
      }
    } catch (err) {
      console.error('Failed to update read status:', err);
      onRead?.(post.id, !newReadState);
      
      if (!isOffline) {
        try {
          await queuePendingRead(post.id, newReadState);
        } catch (e) {
          console.error('Failed to queue:', e);
        }
      }
    } finally {
      setIsUpdating(false);
    }
  }, [post.id, post.is_read, isOffline, isUpdating, onRead]);

  const handleOpenLink = useCallback(() => {
    if (post.source_url) {
      window.open(post.source_url, '_blank', 'noopener,noreferrer');
      if (!post.is_read) handleToggleRead();
    }
  }, [post.source_url, post.is_read, handleToggleRead]);

  const sourceInfo = getSourceInfo(post.source_name);
  const score = post.classifier_score ? Math.round(post.classifier_score * 100) : null;

  return (
    <Card className={`transition-opacity ${post.is_read ? 'opacity-50' : ''}`}>
      <CardHeader className="p-4">
        <div className="flex items-start gap-3">
          <CardTitle 
            className="flex-1 cursor-pointer text-sm font-medium leading-snug hover:text-primary sm:text-base"
            onClick={handleOpenLink}
          >
            {post.title}
          </CardTitle>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={handleToggleRead}
            disabled={isUpdating}
            aria-label={post.is_read ? 'Mark as unread' : 'Mark as read'}
          >
            {post.is_read ? (
              <Check className="h-4 w-4 text-primary" />
            ) : (
              <Circle className="h-4 w-4" />
            )}
          </Button>
        </div>
      </CardHeader>

      <CardFooter className="flex items-center gap-3 border-t px-4 py-3">
        <Badge variant={sourceInfo.color} className="text-xs">
          {sourceInfo.label}
        </Badge>

        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          {formatRelativeTime(post.published_at)}
        </span>

        {score !== null && (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <TrendingUp className="h-3 w-3" />
            {score}%
          </span>
        )}

        <Button
          variant="ghost"
          size="sm"
          className="ml-auto h-7 gap-1.5 px-2 text-xs"
          onClick={handleOpenLink}
        >
          <ExternalLink className="h-3 w-3" />
          Open
        </Button>
      </CardFooter>
    </Card>
  );
}
