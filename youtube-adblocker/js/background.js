// Background service worker for YouTube AdBlocker
chrome.runtime.onInstalled.addListener(() => {
  console.log('YouTube AdBlocker extension installed');
  
  // Initialize default settings
  chrome.storage.sync.set({
    adBlockerEnabled: true,
    adsBlocked: 0
  });
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'updateAdsBlocked') {
    // Update badge with ads blocked count
    updateBadge(message.count);
    sendResponse({ success: true });
  }
  
  if (message.action === 'getTabId') {
    sendResponse({ tabId: sender.tab?.id });
  }
});

// Update extension badge with ads blocked count
async function updateBadge(count) {
  try {
    await chrome.action.setBadgeText({
      text: count > 0 ? count.toString() : ''
    });
    
    await chrome.action.setBadgeBackgroundColor({
      color: '#ff0000'
    });
  } catch (error) {
    console.log('Could not update badge:', error);
  }
}

// Listen for tab updates to reset badge and inject content script
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url && tab.url.includes('youtube.com')) {
    // Reset badge for new YouTube pages
    try {
      const result = await chrome.storage.sync.get(['adsBlocked']);
      updateBadge(result.adsBlocked || 0);
    } catch (error) {
      console.log('Could not reset badge:', error);
    }
  }
});

// Handle extension state changes
chrome.storage.onChanged.addListener((changes, namespace) => {
  if (namespace === 'sync') {
    if (changes.adsBlocked) {
      updateBadge(changes.adsBlocked.newValue || 0);
    }
    
    if (changes.adBlockerEnabled) {
      // Update icon based on enabled state
      const iconPath = changes.adBlockerEnabled.newValue ? {
        '16': 'icons/icon16.png',
        '32': 'icons/icon32.png',
        '48': 'icons/icon48.png',
        '128': 'icons/icon128.png'
      } : {
        '16': 'icons/icon16-disabled.png',
        '32': 'icons/icon32-disabled.png',
        '48': 'icons/icon48-disabled.png',
        '128': 'icons/icon128-disabled.png'
      };
      
      chrome.action.setIcon({ path: iconPath }).catch(err => {
        console.log('Could not update icon:', err);
      });
    }
  }
});

// Initialize badge on startup
chrome.storage.sync.get(['adsBlocked'], (result) => {
  updateBadge(result.adsBlocked || 0);
});