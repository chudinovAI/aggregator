import { useState, useEffect, useCallback } from 'react';
import { Settings, WifiOff, Download, X } from 'lucide-react';
import PostList from './components/PostList';
import SettingsPanel from './components/SettingsPanel';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';
import { setUserId } from './api/client';
import { getPendingReads, removePendingRead } from './utils/db';
import { markPostRead, markPostUnread } from './api/client';

/**
 * Initialize Telegram Web App
 * Returns user info from Telegram or localStorage fallback
 */
function initTelegramWebApp() {
  const tg = window.Telegram?.WebApp;
  
  if (tg) {
    tg.expand();
    tg.ready();
    
    const user = tg.initDataUnsafe?.user;
    if (user?.id) {
      setUserId(user.id.toString());
      console.log('Telegram user:', user.id);
      return { tg, user, userId: user.id.toString() };
    }
  }
  
  // Fallback: check localStorage for previously saved user
  const savedUserId = localStorage.getItem('telegram_id');
  if (savedUserId) {
    console.log('Using saved user:', savedUserId);
    return { tg, user: null, userId: savedUserId };
  }
  
  return { tg: tg || null, user: null, userId: null };
}

/**
 * Sync pending offline actions
 */
async function syncPendingReads() {
  try {
    const pending = await getPendingReads();
    for (const item of pending) {
      try {
        if (item.read) {
          await markPostRead(item.postId);
        } else {
          await markPostUnread(item.postId);
        }
        await removePendingRead(item.postId);
      } catch (err) {
        console.error('Sync failed:', item.postId, err);
      }
    }
  } catch (err) {
    console.error('Sync pending reads failed:', err);
  }
}

/**
 * Main App
 */
export default function App() {
  const [isOffline, setIsOffline] = useState(!navigator.onLine);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [telegramApp, setTelegramApp] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // Initialize Telegram
  useEffect(() => {
    const { tg } = initTelegramWebApp();
    setTelegramApp(tg);
    
    if (tg) {
      tg.BackButton.onClick(() => {
        if (settingsOpen) {
          setSettingsOpen(false);
        } else {
          tg.close();
        }
      });
    }
    
    return () => {
      if (tg) {
        tg.BackButton.offClick();
      }
    };
  }, []);

  // Telegram Back Button
  useEffect(() => {
    if (telegramApp) {
      if (settingsOpen) {
        telegramApp.BackButton.show();
      } else {
        telegramApp.BackButton.hide();
      }
    }
  }, [settingsOpen, telegramApp]);

  // Online/Offline
  useEffect(() => {
    const handleOnline = () => {
      setIsOffline(false);
      syncPendingReads();
    };
    const handleOffline = () => setIsOffline(true);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Sync on load
  useEffect(() => {
    if (!isOffline) syncPendingReads();
  }, []);

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex h-12 items-center justify-between px-3 sm:h-14 sm:px-4">
          <h1 className="text-base font-semibold sm:text-lg">News</h1>
          <div className="flex items-center gap-1.5 sm:gap-2">
            {isOffline && (
              <Badge variant="secondary" className="h-6 gap-1 px-2 text-xs">
                <WifiOff className="h-3 w-3" />
                <span className="hidden sm:inline">Offline</span>
              </Badge>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 sm:h-9 sm:w-9"
              onClick={() => setSettingsOpen(true)}
              aria-label="Settings"
            >
              <Settings className="h-4 w-4 sm:h-5 sm:w-5" />
            </Button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 px-2 py-3 sm:px-4 sm:py-4">
        <PostList isOffline={isOffline} refreshKey={refreshKey} />
      </main>

      {/* Settings */}
      <SettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSettingsSaved={() => setRefreshKey((k) => k + 1)}
        isOffline={isOffline}
      />

      {/* Install Prompt */}
      <InstallPrompt />
    </div>
  );
}

/**
 * PWA Install Prompt
 */
function InstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    const handler = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setTimeout(() => setShowPrompt(true), 5000);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const handleInstall = async () => {
    if (deferredPrompt) {
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      if (outcome === 'accepted') setShowPrompt(false);
      setDeferredPrompt(null);
    }
  };

  if (!showPrompt || !deferredPrompt) return null;

  return (
    <div className="fixed bottom-3 left-3 right-3 flex items-center justify-between gap-2 rounded-lg border bg-card p-3 shadow-lg sm:bottom-4 sm:left-4 sm:right-4 sm:p-4">
      <span className="text-xs sm:text-sm">Install for better experience</span>
      <div className="flex gap-1.5">
        <Button size="sm" className="h-7 text-xs sm:h-8" onClick={handleInstall}>
          <Download className="mr-1 h-3 w-3" />
          Install
        </Button>
        <Button variant="ghost" size="icon" className="h-7 w-7 sm:h-8 sm:w-8" onClick={() => setShowPrompt(false)}>
          <X className="h-3 w-3" />
        </Button>
      </div>
    </div>
  );
}
