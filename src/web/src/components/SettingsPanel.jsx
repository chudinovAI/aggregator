import { useState, useEffect, useCallback } from 'react';
import { X, Save, Loader2, Check, AlertCircle } from 'lucide-react';
import TopicManager from './TopicManager';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select } from './ui/select';
import { getSettings, updateSettings, setUserId } from '../api/client';
import { saveSettings, getCachedSettings } from '../utils/db';

const AVAILABLE_SOURCES = [
  { id: 'reddit', name: 'Reddit', description: 'Tech subreddits' },
  { id: 'hackernews', name: 'Hacker News', description: 'YC news' },
];

const PERIOD_OPTIONS = [
  { value: '1d', label: '1 day' },
  { value: '3d', label: '3 days' },
  { value: '7d', label: '1 week' },
  { value: '14d', label: '2 weeks' },
  { value: '30d', label: '1 month' },
];

export default function SettingsPanel({ isOpen, onClose, onSettingsSaved, isOffline = false }) {
  const [settings, setSettings] = useState({
    topics: [],
    sources: ['reddit', 'hackernews'],
    period: '7d',
  });
  const [telegramId, setTelegramId] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    async function loadSettings() {
      setLoading(true);
      setError(null);

      try {
        const storedId = localStorage.getItem('telegram_id') || '';
        setTelegramId(storedId);

        if (isOffline) {
          const cached = await getCachedSettings();
          if (cached) setSettings(cached);
        } else if (storedId) {
          const response = await getSettings();
          if (response.preferences) {
            setSettings(response.preferences);
            await saveSettings(response.preferences);
          }
        }
      } catch (err) {
        console.error('Load settings failed:', err);
        const cached = await getCachedSettings();
        if (cached) {
          setSettings(cached);
          setError('Using cached settings');
        } else {
          setError('Failed to load settings');
        }
      } finally {
        setLoading(false);
      }
    }

    if (isOpen) loadSettings();
  }, [isOpen, isOffline]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      setUserId(telegramId);

      if (isOffline) {
        await saveSettings(settings);
        setSuccess('Saved locally');
      } else {
        await updateSettings(settings);
        await saveSettings(settings);
        setSuccess('Saved');
      }

      // Notify parent to refresh posts
      if (onSettingsSaved) {
        onSettingsSaved();
      }

      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      console.error('Save failed:', err);
      await saveSettings(settings);
      setError('Saved locally');
    } finally {
      setSaving(false);
    }
  }, [settings, telegramId, isOffline]);

  const handleTopicsChange = useCallback((newTopics) => {
    setSettings((prev) => ({ ...prev, topics: newTopics }));
  }, []);

  const handleSourceToggle = useCallback((sourceId) => {
    setSettings((prev) => {
      const sources = prev.sources || [];
      if (sources.includes(sourceId)) {
        return { ...prev, sources: sources.filter((s) => s !== sourceId) };
      } else {
        return { ...prev, sources: [...sources, sourceId] };
      }
    });
  }, []);

  const handlePeriodChange = useCallback((e) => {
    setSettings((prev) => ({
      ...prev,
      period: e.target.value,
    }));
  }, []);

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div 
        className="fixed inset-y-0 right-0 w-full border-l bg-background shadow-xl sm:max-w-sm"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex h-12 items-center justify-between border-b px-3 sm:h-14 sm:px-4">
          <h2 className="text-base font-semibold sm:text-lg">Settings</h2>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Content */}
        <div className="h-[calc(100vh-3rem)] overflow-y-auto p-3 sm:h-[calc(100vh-3.5rem)] sm:p-4">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="space-y-5">
              {/* Telegram ID */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium sm:text-sm">Telegram ID</label>
                <Input
                  type="text"
                  placeholder="Your Telegram ID"
                  value={telegramId}
                  onChange={(e) => setTelegramId(e.target.value)}
                  className="h-9"
                />
                <p className="text-xs text-muted-foreground">
                  Get from @userinfobot
                </p>
              </div>

              {/* Topics */}
              <TopicManager
                topics={settings.topics || []}
                onChange={handleTopicsChange}
                disabled={saving}
              />

              {/* Sources */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium sm:text-sm">Sources</label>
                <div className="space-y-1.5">
                  {AVAILABLE_SOURCES.map((source) => (
                    <label 
                      key={source.id} 
                      className="flex cursor-pointer items-center gap-2.5 rounded-md border bg-card p-2.5 transition-colors hover:bg-accent"
                    >
                      <input
                        type="checkbox"
                        checked={(settings.sources || []).includes(source.id)}
                        onChange={() => handleSourceToggle(source.id)}
                        disabled={saving}
                        className="h-4 w-4 rounded border-input"
                      />
                      <div className="flex-1">
                        <div className="text-sm font-medium">{source.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {source.description}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Period */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium sm:text-sm">Period</label>
                <Select
                  value={settings.period || '7d'}
                  onChange={handlePeriodChange}
                  disabled={saving}
                  className="h-9"
                >
                  {PERIOD_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </Select>
              </div>

              {/* Messages */}
              {error && (
                <div className="flex items-center gap-2 rounded-md border border-destructive/50 bg-destructive/10 p-2.5 text-xs text-destructive">
                  <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                  {error}
                </div>
              )}
              {success && (
                <div className="flex items-center gap-2 rounded-md border border-primary/50 bg-primary/10 p-2.5 text-xs text-primary">
                  <Check className="h-3.5 w-3.5 shrink-0" />
                  {success}
                </div>
              )}

              {/* Save */}
              <Button
                className="h-9 w-full text-sm"
                onClick={handleSave}
                disabled={saving}
              >
                {saving ? (
                  <>
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="mr-1.5 h-3.5 w-3.5" />
                    Save
                  </>
                )}
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
