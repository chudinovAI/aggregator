import { useState, useCallback } from 'react';
import { X, Plus } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Input } from './ui/input';

const SUGGESTED_TOPICS = [
  'technology',
  'programming',
  'javascript',
  'python',
  'machine learning',
  'ai',
  'startups',
  'golang',
  'rust',
  'devops',
  'web development',
  'mobile',
  'data science',
  'cloud',
  'security',
];

export default function TopicManager({ topics = [], onChange, disabled = false }) {
  const [inputValue, setInputValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  const normalizedTopics = topics.map((t) => t.toLowerCase().trim());

  const handleAddTopic = useCallback((topic) => {
    const normalized = topic.toLowerCase().trim();
    if (normalized && !normalizedTopics.includes(normalized)) {
      onChange([...topics, normalized]);
    }
    setInputValue('');
    setShowSuggestions(false);
  }, [topics, normalizedTopics, onChange]);

  const handleRemoveTopic = useCallback((topicToRemove) => {
    onChange(topics.filter((t) => t.toLowerCase() !== topicToRemove.toLowerCase()));
  }, [topics, onChange]);

  const handleInputKeyDown = useCallback((e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (inputValue.trim()) handleAddTopic(inputValue);
    }
  }, [inputValue, handleAddTopic]);

  const filteredSuggestions = SUGGESTED_TOPICS.filter(
    (t) => t.includes(inputValue.toLowerCase()) && !normalizedTopics.includes(t)
  );

  return (
    <div className="space-y-2">
      <label className="text-xs font-medium sm:text-sm">Topics</label>
      
      {/* Current topics */}
      <div className="flex flex-wrap gap-1.5">
        {topics.length === 0 && (
          <span className="text-xs text-muted-foreground">No topics</span>
        )}
        {topics.map((topic) => (
          <Badge key={topic} variant="secondary" className="gap-1 py-1 pr-1 text-xs">
            {topic}
            <button
              className="ml-0.5 rounded-full p-0.5 hover:bg-background/50"
              onClick={() => handleRemoveTopic(topic)}
              disabled={disabled}
            >
              <X className="h-2.5 w-2.5" />
            </button>
          </Badge>
        ))}
      </div>

      {/* Input */}
      <div className="relative">
        <div className="flex gap-1.5">
          <Input
            type="text"
            placeholder="Add topic..."
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              setShowSuggestions(true);
            }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
            onKeyDown={handleInputKeyDown}
            disabled={disabled}
            className="h-8 flex-1 text-sm"
          />
          <Button
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => handleAddTopic(inputValue)}
            disabled={disabled || !inputValue.trim()}
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Suggestions */}
        {showSuggestions && filteredSuggestions.length > 0 && (
          <div className="absolute left-0 right-0 top-full z-10 mt-1 max-h-36 overflow-auto rounded-md border bg-card shadow-lg">
            {filteredSuggestions.slice(0, 6).map((s) => (
              <button
                key={s}
                className="w-full px-2.5 py-1.5 text-left text-xs hover:bg-accent sm:text-sm"
                onClick={() => handleAddTopic(s)}
                type="button"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Quick add */}
      <div className="flex flex-wrap gap-1">
        {SUGGESTED_TOPICS.filter((t) => !normalizedTopics.includes(t))
          .slice(0, 4)
          .map((topic) => (
            <Button
              key={topic}
              variant="outline"
              size="sm"
              className="h-6 gap-0.5 px-1.5 text-xs"
              onClick={() => handleAddTopic(topic)}
              disabled={disabled}
            >
              <Plus className="h-2.5 w-2.5" />
              {topic}
            </Button>
          ))}
      </div>
    </div>
  );
}
