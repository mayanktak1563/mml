# YouTube AdBlocker Chrome Extension

A lightweight and effective Chrome extension that blocks all types of advertisements on YouTube, including pre-roll, mid-roll, post-roll, banner, overlay, and sidebar ads.

## Features

- ✅ Blocks pre-roll video ads (ads before videos)
- ✅ Blocks mid-roll video ads (ads during videos) 
- ✅ Blocks post-roll video ads (ads after videos)
- ✅ Blocks banner advertisements
- ✅ Blocks overlay ads on videos
- ✅ Blocks homepage and sidebar advertisements
- ✅ Blocks YouTube Shorts ads
- ✅ Blocks sponsored content
- ✅ Simple toggle button to enable/disable blocking
- ✅ Ads blocked counter
- ✅ Real-time ad detection and removal
- ✅ Manifest V3 compliant
- ✅ Lightweight and fast

## Installation

### From Source (For Development)

1. Clone or download this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" in the top right corner
4. Click "Load unpacked" and select the `youtube-adblocker` folder
5. The extension will be installed and ready to use

### Icons Setup

Before using the extension, you need to add icon files to the `icons/` directory:
- `icon16.png` (16x16 pixels)
- `icon32.png` (32x32 pixels) 
- `icon48.png` (48x48 pixels)
- `icon128.png` (128x128 pixels)

See `icons/README.md` for more details on creating icons.

## Usage

1. Navigate to any YouTube page
2. Click the YouTube AdBlocker extension icon in the Chrome toolbar
3. Use the toggle switch to enable/disable ad blocking
4. View the number of ads blocked in the popup
5. Enjoy ad-free YouTube!

## How It Works

The extension uses several techniques to block YouTube ads:

### Content Script Injection
- Injects JavaScript into YouTube pages to identify and remove ad elements
- Uses comprehensive CSS selectors to target various ad types
- Removes ad containers, banners, overlays, and sponsored content

### MutationObserver
- Monitors DOM changes to catch dynamically loaded ads
- Removes new ad elements as they appear on the page
- Ensures ads are blocked even when YouTube updates its interface

### Video Ad Handling
- Detects video ads by monitoring for ad indicators
- Automatically clicks skip buttons when available
- Fast-forwards through non-skippable ads when possible

### Real-time Monitoring
- Continuously scans for new ad elements every 2 seconds
- Adapts to YouTube's changing ad delivery methods
- Maintains blocking effectiveness as YouTube updates

## Technical Details

### Manifest V3 Compliance
- Uses service worker instead of background pages
- Implements proper permissions and host permissions
- Follows Chrome extension best practices

### File Structure
```
youtube-adblocker/
├── manifest.json          # Extension manifest (Manifest V3)
├── popup.html             # Extension popup interface
├── css/
│   ├── popup.css          # Popup styling
│   └── content.css        # Content script CSS for ad hiding
├── js/
│   ├── popup.js           # Popup functionality
│   ├── content.js         # Main ad blocking logic
│   └── background.js      # Background service worker
├── icons/
│   ├── README.md          # Icon setup instructions
│   └── [icon files]       # Extension icons (16, 32, 48, 128px)
└── README.md              # This file
```

### Permissions
- `activeTab`: Access to the current active tab for ad blocking
- `storage`: Save extension settings and ads blocked count
- `*://*.youtube.com/*`: Access to all YouTube pages
- `*://*.googlevideo.com/*`: Access to YouTube video content

## Privacy

This extension:
- ✅ Does NOT collect any personal data
- ✅ Does NOT track your browsing habits  
- ✅ Does NOT send data to external servers
- ✅ Only operates on YouTube pages
- ✅ Stores settings locally in Chrome storage

## Troubleshooting

### Extension Not Working
1. Make sure you're on a YouTube page
2. Check that the extension is enabled in Chrome extensions
3. Try refreshing the YouTube page
4. Ensure the toggle is set to "enabled" in the popup

### Some Ads Still Showing
1. YouTube frequently updates their ad system
2. Try refreshing the page
3. Make sure the extension is updated to the latest version
4. Some ads may take a moment to be detected and removed

### Performance Issues
1. The extension is designed to be lightweight
2. If you experience issues, try disabling other extensions
3. Clear browser cache and restart Chrome

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Test your changes thoroughly on different YouTube pages
4. Submit a pull request with a clear description

### Testing Checklist
- [ ] Test on YouTube homepage
- [ ] Test on video watch pages  
- [ ] Test on YouTube Shorts
- [ ] Test with different video types (music, regular videos)
- [ ] Test toggle functionality
- [ ] Test ads blocked counter
- [ ] Verify no YouTube functionality is broken

## Known Limitations

- YouTube continuously updates their ad delivery system
- Some new ad types may temporarily show until the extension is updated
- Very new or experimental ad formats might not be immediately blocked
- The extension only works on YouTube (by design)

## Version History

### v1.0.0
- Initial release
- Blocks all major YouTube ad types
- Popup interface with toggle
- Ads blocked counter
- Manifest V3 compliance
- Real-time ad detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This extension is for educational purposes. Users should comply with YouTube's terms of service and consider supporting content creators through other means.