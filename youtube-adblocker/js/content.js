// Content script for YouTube ad blocking
(function() {
  'use strict';
  
  let isEnabled = true;
  let adsBlocked = 0;
  let observer = null;
  
  // YouTube ad selectors - comprehensive list targeting various ad elements
  const AD_SELECTORS = [
    // Video ads
    '.video-ads',
    '.ytp-ad-module',
    '.ytp-ad-overlay-container',
    '.ytp-ad-player-overlay',
    '.ytp-ad-skip-button-container',
    '.ytp-ad-text',
    '.ytp-ad-preview-container',
    'ytd-display-ad-renderer',
    'ytd-video-masthead-ad-primary-video-renderer',
    'ytd-in-feed-ad-layout-renderer',
    'ytd-ad-slot-renderer',
    'yt-mealbar-promo-renderer',
    '.ytd-mealbar-promo-renderer',
    
    // Banner and sidebar ads
    'ytd-banner-promo-renderer',
    '.ytd-banner-promo-renderer',
    'ytd-brand-video-singleton-renderer',
    'ytd-brand-video-shelf-renderer',
    'ytd-promoted-sparkles-web-renderer',
    'ytd-promoted-video-renderer',
    'ytd-compact-promoted-video-renderer',
    'ytd-display-ad-renderer',
    'ytd-promoted-sparkles-text-search-renderer',
    '#masthead-ad',
    '.masthead-ad-control',
    
    // Overlay ads
    '.ytp-ad-overlay-container',
    '.ytp-ce-covering-overlay',
    '.ytp-ad-overlay-close-button',
    
    // Home page ads
    'ytd-rich-item-renderer[is-ad]',
    'ytd-video-renderer[is-ad]',
    'ytd-compact-video-renderer[is-ad]',
    
    // YouTube Shorts ads
    'ytd-ad-slot-renderer',
    'ytd-player-legacy-desktop-watch-ads-renderer',
    
    // Generic ad containers
    '[id*="google_ads"]',
    '[class*="google-ads"]',
    '[data-ad-slot]',
    '.advertisement',
    '.ads',
    '.ad-container',
    '[id*="adsystem"]',
    
    // YouTube Premium promotions
    'ytd-premium-promo-renderer',
    'ytd-statement-banner-renderer',
    
    // Sponsored content
    '[aria-label*="Sponsored"]',
    '[title*="Sponsored"]',
    'span:contains("Sponsored")',
    'span:contains("Ad")',
    
    // Additional video ad elements
    '.ytp-ad-button',
    '.ytp-ad-duration-remaining',
    '.ytp-ad-visit-advertiser-button',
    '.ytp-flyout-cta',
    '.ytp-videowall-still-info-content',
    
    // Mobile specific
    '.mobile-topbar-header-ad-slot',
    '.ytm-promoted-video-renderer',
    '.ytm-compact-promoted-video-renderer'
  ];
  
  // Initialize the ad blocker
  async function initialize() {
    // Get saved state
    try {
      const result = await chrome.storage.sync.get(['adBlockerEnabled', 'adsBlocked']);
      isEnabled = result.adBlockerEnabled !== false;
      adsBlocked = result.adsBlocked || 0;
    } catch (error) {
      console.log('Could not load ad blocker state:', error);
    }
    
    if (isEnabled) {
      startBlocking();
    }
    
    // Listen for messages from popup
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.action === 'toggleAdBlocker') {
        isEnabled = message.enabled;
        if (isEnabled) {
          startBlocking();
        } else {
          stopBlocking();
        }
        sendResponse({ success: true });
      }
    });
  }
  
  function startBlocking() {
    // Remove existing ads immediately
    removeAds();
    
    // Set up MutationObserver to catch dynamically loaded ads
    if (observer) {
      observer.disconnect();
    }
    
    observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              removeAdsFromElement(node);
            }
          });
        }
      });
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
    
    // Also block video ads by intercepting video events
    blockVideoAds();
    
    console.log('YouTube AdBlocker: Started blocking ads');
  }
  
  function stopBlocking() {
    if (observer) {
      observer.disconnect();
      observer = null;
    }
    console.log('YouTube AdBlocker: Stopped blocking ads');
  }
  
  function removeAds() {
    AD_SELECTORS.forEach(selector => {
      try {
        const elements = document.querySelectorAll(selector);
        elements.forEach(element => {
          if (element && element.parentNode) {
            element.style.display = 'none';
            element.remove();
            incrementAdsBlocked();
          }
        });
      } catch (error) {
        // Some selectors might be invalid, ignore errors
      }
    });
    
    // Remove elements with ad-related attributes
    const adElements = document.querySelectorAll('[data-ad-slot], [id*="google_ads"], [class*="google-ads"]');
    adElements.forEach(element => {
      if (element && element.parentNode) {
        element.style.display = 'none';
        element.remove();
        incrementAdsBlocked();
      }
    });
    
    // Remove sponsored content
    const sponsoredElements = document.querySelectorAll('*');
    sponsoredElements.forEach(element => {
      const text = element.textContent || '';
      if (text.includes('Sponsored') || text.includes('Ad •') || 
          element.getAttribute('aria-label')?.includes('Sponsored')) {
        const container = element.closest('ytd-video-renderer, ytd-compact-video-renderer, ytd-rich-item-renderer');
        if (container) {
          container.style.display = 'none';
          container.remove();
          incrementAdsBlocked();
        }
      }
    });
  }
  
  function removeAdsFromElement(element) {
    if (!isEnabled) return;
    
    AD_SELECTORS.forEach(selector => {
      try {
        if (element.matches && element.matches(selector)) {
          element.style.display = 'none';
          element.remove();
          incrementAdsBlocked();
          return;
        }
        
        const childElements = element.querySelectorAll(selector);
        childElements.forEach(childElement => {
          if (childElement && childElement.parentNode) {
            childElement.style.display = 'none';
            childElement.remove();
            incrementAdsBlocked();
          }
        });
      } catch (error) {
        // Ignore selector errors
      }
    });
  }
  
  function blockVideoAds() {
    // Skip video ads by detecting and skipping them
    const skipVideoAds = () => {
      const video = document.querySelector('video');
      if (!video) return;
      
      // Check for ad indicators
      const adIndicators = document.querySelectorAll('.ytp-ad-text, .ytp-ad-duration-remaining, .ytp-ad-skip-button');
      if (adIndicators.length > 0) {
        // This is an ad, try to skip it
        const skipButton = document.querySelector('.ytp-ad-skip-button, .ytp-skip-ad-button');
        if (skipButton && skipButton.offsetParent !== null) {
          skipButton.click();
          incrementAdsBlocked();
        } else {
          // If no skip button, try to fast-forward the ad
          if (video.currentTime < video.duration - 0.1) {
            video.currentTime = video.duration - 0.1;
          }
        }
      }
    };
    
    // Check for video ads periodically
    setInterval(skipVideoAds, 500);
    
    // Also listen for video events
    document.addEventListener('video', skipVideoAds);
  }
  
  function incrementAdsBlocked() {
    adsBlocked++;
    // Save to storage
    chrome.storage.sync.set({ adsBlocked: adsBlocked });
    
    // Notify popup
    try {
      chrome.runtime.sendMessage({
        action: 'updateAdsBlocked',
        count: adsBlocked
      });
    } catch (error) {
      // Popup might not be open
    }
  }
  
  // Run ad removal on page load and when DOM changes
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    initialize();
  }
  
  // Remove ads every few seconds as backup
  setInterval(() => {
    if (isEnabled) {
      removeAds();
    }
  }, 2000);
  
})();