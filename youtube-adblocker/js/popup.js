// Popup JavaScript functionality
document.addEventListener('DOMContentLoaded', async function() {
  const toggle = document.getElementById('adBlockerToggle');
  const statusText = document.getElementById('statusText');
  const status = document.getElementById('status');
  const adsBlockedElement = document.getElementById('adsBlocked');
  
  // Load saved state
  const result = await chrome.storage.sync.get(['adBlockerEnabled', 'adsBlocked']);
  const isEnabled = result.adBlockerEnabled !== false; // Default to true
  const adsBlocked = result.adsBlocked || 0;
  
  // Set initial state
  toggle.checked = isEnabled;
  adsBlockedElement.textContent = adsBlocked;
  updateStatus(isEnabled);
  
  // Add toggle event listener
  toggle.addEventListener('change', async function() {
    const enabled = toggle.checked;
    
    // Save state
    await chrome.storage.sync.set({ adBlockerEnabled: enabled });
    
    // Update UI
    updateStatus(enabled);
    
    // Send message to content script
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && tab.url && tab.url.includes('youtube.com')) {
        await chrome.tabs.sendMessage(tab.id, {
          action: 'toggleAdBlocker',
          enabled: enabled
        });
      }
    } catch (error) {
      console.log('Could not send message to content script:', error);
    }
  });
  
  function updateStatus(enabled) {
    if (enabled) {
      statusText.textContent = 'AdBlocker is enabled';
      status.className = 'status enabled';
    } else {
      statusText.textContent = 'AdBlocker is disabled';
      status.className = 'status disabled';
    }
  }
  
  // Listen for updates from content script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'updateAdsBlocked') {
      adsBlockedElement.textContent = message.count;
    }
  });
  
  // Update ads blocked count from storage periodically
  setInterval(async () => {
    const result = await chrome.storage.sync.get(['adsBlocked']);
    const adsBlocked = result.adsBlocked || 0;
    adsBlockedElement.textContent = adsBlocked;
  }, 1000);
});